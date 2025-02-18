from __future__ import annotations

import logging
import typing as t

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.executor import Executor
from ragas.run_config import RunConfig
import warnings

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.cost import TokenUsageParser
    from ragas.llms import BaseRagasLLM
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.evaluation import EvaluationResult
    from ragas.metrics.base import Metric
    from r2r import R2RAsyncClient


logger = logging.getLogger(__name__)


def _process_search_results(search_results: t.Dict[str, t.List]) -> t.List[str]:
    """
    Extracts relevant text from search results while issuing warnings for unsupported result types.

    Parameters
    ----------
    search_results : Dict[str, List]
        A r2r result object of an aggregate search operation.

    Returns
    -------
    List[str]
        A list of extracted text from aggregate search result.
    """
    retrieved_contexts = []

    for key in ["graph_search_results", "context_document_results"]:
        if search_results.get(key) and len(search_results[key]) > 0:
            warnings.warn(
                f"{key} are not included in the aggregated `retrieved_context` for Ragas evaluations."
            )

    for result in search_results.get("chunk_search_results", []):
        text = result.get("text")
        if text:
            retrieved_contexts.append(text)

    for result in search_results.get("web_search_results", []):
        text = result.get("snippet")
        if text:
            retrieved_contexts.append(text)

    return retrieved_contexts


def evaluate(
    r2r_client: R2RAsyncClient,
    dataset: EvaluationDataset,
    metrics: list[Metric],
    search_settings: t.Optional[t.Dict[str, t.Any]] = None,
    rag_generation_config: t.Optional[t.Dict[str,]] = None,
    search_mode: t.Optional[str] = "custom",
    task_prompt_override: t.Optional[str] = None,
    include_title_if_available: t.Optional[bool] = False,
    llm: t.Optional[BaseRagasLLM] = None,
    embeddings: t.Optional[BaseRagasEmbeddings] = None,
    callbacks: t.Optional[Callbacks] = None,
    run_config: t.Optional[RunConfig] = None,
    batch_size: t.Optional[int] = None,
    token_usage_parser: t.Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    show_progress: bool = True,
) -> EvaluationResult:
    column_map = column_map or {}

    # validate and transform dataset
    if dataset is None or not isinstance(dataset, EvaluationDataset):
        raise ValueError("Please provide a dataset that is of type EvaluationDataset")

    exec = Executor(
        desc="Querying Client",
        keep_progress_bar=True,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        batch_size=batch_size,
    )

    # check if multi-turn
    if dataset.is_multi_turn():
        raise NotImplementedError(
            "Multi-turn evaluation is not implemented yet. Please do raise an issue on GitHub if you need this feature and we will prioritize it"
        )
    samples = t.cast(t.List[SingleTurnSample], dataset.samples)

    # get query and make jobs
    queries = [sample.user_input for sample in samples]
    for i, q in enumerate(queries):
        exec.submit(
            r2r_client.retrieval.rag,
            query=q,
            rag_generation_config=rag_generation_config,
            search_mode=search_mode,
            search_settings=search_settings,
            task_prompt_override=task_prompt_override,
            include_title_if_available=include_title_if_available,
            name=f"query-{i}",
        )

    # get responses and retrieved contexts
    responses: t.List[str] = []
    retrieved_contexts: t.List[t.List[str]] = []
    results = exec.results()

    for r in results:
        responses.append(r.results.generated_answer)
        retrieved_contexts.append(
            _process_search_results(r.results.search_results.as_dict())
        )

    # append the extra information to the dataset
    for i, sample in enumerate(samples):
        sample.response = responses[i]
        sample.retrieved_contexts = retrieved_contexts[i]

    results = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=raise_exceptions,
        callbacks=callbacks,
        show_progress=show_progress,
        run_config=run_config or RunConfig(),
        token_usage_parser=token_usage_parser,
    )

    return results


def transform_to_ragas_dataset(
    user_inputs: t.Optional[t.List[str]] = None,
    r2r_responses: t.Optional[t.List] = None,
    reference_contexts: t.Optional[t.List[str]] = None,
    references: t.Optional[t.List[str]] = None,
    rubrics: t.Optional[t.List[t.Dict[str, str]]] = None,
) -> EvaluationDataset:
    """
    Converts input data into a RAGAS EvaluationDataset, ensuring flexibility
    for cases where only some lists are provided.

    Parameters
    ----------
    user_inputs : Optional[List[str]]
        List of user queries.
    r2r_responses : Optional[List]
        List of responses from the R2R client.
    reference_contexts : Optional[List[str]]
        List of reference contexts.
    references : Optional[List[str]]
        List of reference answers.
    rubrics : Optional[List[Dict[str, str]]]
        List of evaluation rubrics.

    Returns
    -------
    EvaluationDataset
        A dataset containing structured evaluation samples.

    Raises
    ------
    ValueError
        If provided lists (except None ones) do not have the same length.
    """

    # Collect only the non-None lists
    provided_lists = {
        "user_inputs": user_inputs or [],
        "r2r_responses": r2r_responses or [],
        "reference_contexts": reference_contexts or [],
        "references": references or [],
        "rubrics": rubrics or [],
    }

    # Find the maximum length among provided lists
    max_len = max(len(lst) for lst in provided_lists.values())

    # Ensure all provided lists have the same length
    for key, lst in provided_lists.items():
        if lst and len(lst) != max_len:
            raise ValueError(f"Inconsistent length for {key}: expected {max_len}, got {len(lst)}")

    # Create samples while handling missing values
    samples = []
    for i in range(max_len):
        sample = SingleTurnSample(
            user_input=user_inputs[i] if user_inputs else None,
            response=(r2r_responses[i].results.generated_answer if r2r_responses else None),
            retrieved_contexts=(
                _process_search_results(r2r_responses[i].results.search_results.as_dict())
                if r2r_responses else None
            ),
            reference_contexts=reference_contexts[i] if reference_contexts else None,
            reference=references[i] if references else None,
            rubric=rubrics[i] if rubrics else None,
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)
