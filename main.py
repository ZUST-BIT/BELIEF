"""MEDAR-QA example entry point."""

from medar_pipeline import run_pipeline


def main() -> None:
    """Run one example medical multiple-choice question."""
    question = """A 20-year-old man comes to the physician because of worsening gait unsteadiness and bilateral hearing loss for 1 month. He has had intermittent tingling sensations on both cheeks over this time period. He has no history of serious medical illness and takes no medications. Audiometry shows bilateral sensorineural hearing loss. Genetic evaluation shows a mutation of a tumor suppressor gene on chromosome 22 that encodes merlin. This patient is at increased risk for which of the following conditions?
                "A": "Renal cell carcinoma",
                "B": "Meningioma",
                "C": "Astrocytoma",
                "D": "Vascular malformations"
    """

    run_pipeline(
        question=question,
        context="",
        task_mode="SELECTION",
        enable_direct_llm_branch=True,
    )


if __name__ == "__main__":
    main()
