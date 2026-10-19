"""PyKEEN training stage for KG embedding models."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, read_json, write_json


KEY_METRICS = (
    "tail.realistic.inverse_harmonic_mean_rank",
    "tail.realistic.hits_at_1",
    "tail.realistic.hits_at_3",
    "tail.realistic.hits_at_10",
)

MRR_FALLBACK_KEYS = (
    "tail.realistic.inverse_harmonic_mean_rank",
    "tail.realistic.mean_reciprocal_rank",
    "tail.realistic.adjusted_inverse_harmonic_mean_rank",
    "both.realistic.inverse_harmonic_mean_rank",
    "both.realistic.mean_reciprocal_rank",
    "both.realistic.adjusted_inverse_harmonic_mean_rank",
)


def _select_device(device_preference, torch_module):
    if device_preference == "cpu":
        return "cpu"
    if device_preference == "cuda":
        return "cuda"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _extract_numeric_metrics(flat_metrics):
    extracted = {}
    for key, value in flat_metrics.items():
        try:
            extracted[key] = float(value)
        except Exception:
            continue
    return extracted


def _resolve_mrr(metrics):
    for key in MRR_FALLBACK_KEYS:
        if key in metrics:
            return float(metrics[key])
    return 0.0


def _extract_entity_embeddings(model, entity_count, device_name, torch_module):
    entity_representations = model.entity_representations[0]
    if hasattr(entity_representations, "weight"):
        return entity_representations.weight.detach().cpu().numpy()

    all_indices = torch_module.arange(entity_count)
    if device_name == "cuda":
        all_indices = all_indices.cuda()
    return entity_representations(all_indices).detach().cpu().numpy()


def _save_reconstructable_model_artifacts(
    *,
    model,
    training_factory,
    model_dir,
    model_name,
    model_kwargs,
    torch_module,
    numpy_module,
):
    """Save the minimum artifacts needed for later model-level attribution.

    Historical KGML runs retained only the final entity matrix.  That is enough
    for geometry and cross-view analysis, but it cannot recover relation-wise
    RGCN messages.  New runs can opt into these artifacts without serialising a
    version-fragile Python model object.
    """

    state_path = model_dir / "model_state_dict.pt"
    triples_path = model_dir / "training_mapped_triples.npy"
    relation_path = model_dir / "relation_to_id.json"
    torch_module.save(model.state_dict(), str(state_path))
    numpy_module.save(
        str(triples_path),
        training_factory.mapped_triples.detach().cpu().numpy(),
    )
    write_json(
        relation_path,
        {str(key): int(value) for key, value in training_factory.relation_to_id.items()},
    )
    write_json(
        model_dir / "reconstruction_manifest.json",
        {
            "model": str(model_name),
            "model_class": "%s.%s"
            % (model.__class__.__module__, model.__class__.__name__),
            "model_kwargs": model_kwargs,
            "state_dict": state_path.name,
            "training_mapped_triples": triples_path.name,
            "entity_to_id": "entity_to_id.json",
            "relation_to_id": relation_path.name,
            "purpose": (
                "Reconstruct the fitted graph encoder for frozen-weight edge or "
                "relation perturbation; final entity_embeddings.npy alone is "
                "insufficient for RGCN message attribution."
            ),
        },
    )


def train_all_views(config, logger):
    import numpy as np
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    paths = config["paths"]
    training_cfg = config["training"]
    triple_view_dir = Path(paths["intermediate_dir"]) / "triples" / "views"
    processed_root = ensure_dir(Path(paths["processed_dir"]) / "embeddings")

    device_name = _select_device(training_cfg.get("device_preference", "auto"), torch)
    logger.info("PyKEEN training will use device: %s", device_name)
    reuse_completed = bool(training_cfg.get("reuse_completed_results", False))

    summary_rows = []

    for view_name in config.get("views", {}).keys():
        triples_path = triple_view_dir / ("%s.csv" % view_name)
        if not triples_path.exists():
            raise FileNotFoundError("Triple view not found: %s" % triples_path)

        logger.info("Reading triples for view %s from %s", view_name, triples_path)
        triples_df = pd.read_csv(triples_path)
        labeled_triples = triples_df[["head", "relation", "tail"]].astype(str).to_numpy()

        triples_factory = TriplesFactory.from_labeled_triples(labeled_triples)
        split_ratios = training_cfg["split_ratios"]
        training, testing, validation = triples_factory.split(split_ratios, random_state=config["random_seed"])

        view_dir = ensure_dir(processed_root / view_name)
        split_manifest = {
            "view": view_name,
            "training_triples": int(training.num_triples),
            "testing_triples": int(testing.num_triples),
            "validation_triples": int(validation.num_triples),
            "entity_count": int(training.num_entities),
            "relation_count": int(training.num_relations),
        }
        write_json(view_dir / "split_manifest.json", split_manifest)

        for model_name in training_cfg.get("models", []):
            model_dir = ensure_dir(view_dir / model_name)
            logger.info("Training %s on view %s", model_name, view_name)

            model_kwargs = training_cfg.get("model_specific_kwargs", {}).get(model_name, {})
            embedding_path = model_dir / "entity_embeddings.npy"
            entity_path = model_dir / "entity_to_id.json"
            metrics_path = model_dir / "metrics.json"
            manifest_path = model_dir / "run_manifest.json"

            if (
                reuse_completed
                and embedding_path.exists()
                and entity_path.exists()
                and metrics_path.exists()
                and manifest_path.exists()
            ):
                numeric_metrics = read_json(metrics_path)
                run_manifest = read_json(manifest_path)
                training_seconds = float(run_manifest.get("training_seconds", 0.0))
                summary_rows.append({
                    "view": view_name,
                    "model": model_name,
                    "training_minutes": round(training_seconds / 60.0, 4),
                    "mrr": _resolve_mrr(numeric_metrics),
                    "hits_at_1": numeric_metrics.get(KEY_METRICS[1], 0.0),
                    "hits_at_3": numeric_metrics.get(KEY_METRICS[2], 0.0),
                    "hits_at_10": numeric_metrics.get(KEY_METRICS[3], 0.0),
                    "entity_count": int(training.num_entities),
                    "relation_count": int(training.num_relations),
                    "status": "success",
                })
                logger.info(
                    "Reused completed %s on %s | MRR=%.4f | Hits@10=%.4f",
                    model_name,
                    view_name,
                    _resolve_mrr(numeric_metrics),
                    numeric_metrics.get(KEY_METRICS[3], 0.0),
                )
                continue

            start_time = time.time()

            try:
                result = pipeline(
                    model=model_name,
                    training=training,
                    testing=testing,
                    validation=validation,
                    model_kwargs=model_kwargs,
                    training_kwargs=training_cfg.get("training_kwargs", {}),
                    evaluation_kwargs=training_cfg.get("evaluation_kwargs", {}),
                    optimizer=training_cfg.get("optimizer", "adam"),
                    optimizer_kwargs=training_cfg.get("optimizer_kwargs", {}),
                    loss=training_cfg.get("loss", "CrossEntropyLoss"),
                    stopper=training_cfg.get("stopper", "early"),
                    stopper_kwargs=training_cfg.get("stopper_kwargs", {}),
                    random_seed=config["random_seed"],
                    device=device_name,
                )

                training_seconds = time.time() - start_time
                flat_metrics = result.metric_results.to_flat_dict()
                numeric_metrics = _extract_numeric_metrics(flat_metrics)
                entity_to_id = training.entity_to_id
                embeddings = _extract_entity_embeddings(
                    result.model,
                    len(entity_to_id),
                    device_name,
                    torch,
                )

                np.save(str(model_dir / "entity_embeddings.npy"), embeddings)
                write_json(model_dir / "entity_to_id.json", entity_to_id)
                write_json(model_dir / "metrics.json", numeric_metrics)
                saved_model_artifacts = bool(
                    training_cfg.get("save_reconstructable_model_artifacts", False)
                )
                if saved_model_artifacts:
                    _save_reconstructable_model_artifacts(
                        model=result.model,
                        training_factory=training,
                        model_dir=model_dir,
                        model_name=model_name,
                        model_kwargs=model_kwargs,
                        torch_module=torch,
                        numpy_module=np,
                    )
                write_json(
                    model_dir / "run_manifest.json",
                    {
                        "view": view_name,
                        "model": model_name,
                        "training_seconds": training_seconds,
                        "model_kwargs": model_kwargs,
                        "embedding_shape": list(embeddings.shape),
                        "reconstructable_model_artifacts": saved_model_artifacts,
                    },
                )

                summary_rows.append({
                    "view": view_name,
                    "model": model_name,
                    "training_minutes": round(training_seconds / 60.0, 4),
                    "mrr": _resolve_mrr(numeric_metrics),
                    "hits_at_1": numeric_metrics.get(KEY_METRICS[1], 0.0),
                    "hits_at_3": numeric_metrics.get(KEY_METRICS[2], 0.0),
                    "hits_at_10": numeric_metrics.get(KEY_METRICS[3], 0.0),
                    "entity_count": int(training.num_entities),
                    "relation_count": int(training.num_relations),
                    "status": "success",
                })
                logger.info(
                    "Finished %s on %s | MRR=%.4f | Hits@10=%.4f",
                    model_name,
                    view_name,
                    _resolve_mrr(numeric_metrics),
                    numeric_metrics.get(KEY_METRICS[3], 0.0),
                )
            except Exception as exc:
                write_json(
                    model_dir / "error.json",
                    {
                        "view": view_name,
                        "model": model_name,
                        "error": str(exc),
                    },
                )
                summary_rows.append({
                    "view": view_name,
                    "model": model_name,
                    "training_minutes": 0.0,
                    "mrr": 0.0,
                    "hits_at_1": 0.0,
                    "hits_at_3": 0.0,
                    "hits_at_10": 0.0,
                    "entity_count": int(training.num_entities),
                    "relation_count": int(training.num_relations),
                    "status": "failed",
                })
                logger.exception("Training failed for %s on %s", model_name, view_name)

    return pd.DataFrame(summary_rows)
