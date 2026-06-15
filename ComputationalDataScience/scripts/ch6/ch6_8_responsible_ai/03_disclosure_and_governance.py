"""
STAT 418 · Chapter 6.8 — Responsible AI Practices
Script 03: Disclosure statements and a deployment checklist.

Responsible AI is not only about the model - it is about how you communicate and
govern its use. generate_disclosure builds an AI-use disclosure from a structured
record; deployment_checklist emits a go/no-go checklist spanning privacy,
reliability, bias, transparency, and oversight (the Section 6.7 reliability metrics
appear as concrete thresholds here).

Pure Python - no API key required.

Prereqs / run: nothing beyond the standard library.
"""
from __future__ import annotations


def generate_disclosure(ai_uses: list) -> str:
    """Build an AI-use disclosure statement from a list of structured uses."""
    if not ai_uses:
        return "No AI tools were used in this analysis."
    lines = ["AI Disclosure Statement", "",
             "The following AI tools were used in this work:"]
    for use in ai_uses:
        lines.append(f"- {use['task']}: {use['tool']} was used for "
                     f"{use['description']}. {use['human_role']}")
    lines += ["", "All AI-generated outputs were reviewed and validated by the "
              "authors before inclusion."]
    return "\n".join(lines)


def deployment_checklist(task_description: str) -> None:
    """Print a responsible-AI go/no-go checklist for a task."""
    checklist = {
        "Privacy": [
            "Data does not contain unredacted PII",
            "Data handling complies with HIPAA/FERPA/GDPR as applicable",
            "API provider data policies reviewed",
        ],
        "Reliability": [
            "Evaluated on a representative test set (n >= 30)",
            "Accuracy exceeds the task-specific threshold",
            "Consistency measured via test-retest (agreement > 80%)",
            "Calibration assessed (ECE < 0.15)",
        ],
        "Bias": [
            "Tested for differential treatment across demographic groups",
            "No systematic bias detected (or documented and mitigated)",
        ],
        "Transparency": [
            "AI use will be disclosed appropriately",
            "Methodology documented for reproducibility",
            "Limitations clearly stated",
        ],
        "Oversight": [
            "Human review process defined",
            "Escalation path for uncertain or flagged cases",
            "Monitoring plan for the deployed system",
        ],
    }
    print(f"Deployment checklist for: {task_description}")
    print("=" * 52)
    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  [ ] {item}")


def main() -> None:
    print(generate_disclosure([
        {"task": "Data Annotation",
         "tool": "gemma3:12b via Purdue GenAI Studio",
         "description": "sentiment classification of 5,000 customer reviews",
         "human_role": "A random sample of 200 reviews was manually verified "
                       "(Cohen's kappa = 0.78)."},
        {"task": "Text Preprocessing",
         "tool": "Custom Python pipeline",
         "description": "cleaning and chunking raw text data",
         "human_role": "Pipeline validated on 50 representative samples."},
    ]))
    print("\n")
    deployment_checklist("Sentiment annotation of 10,000 customer reviews")


if __name__ == "__main__":
    main()
