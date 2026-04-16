import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
REPOSITORY = os.environ["GITHUB_REPOSITORY"]
PR_NUMBER = os.environ["PR_NUMBER"]
GITHUB_API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

MAX_FILES = 25
MAX_PATCH_CHARS_PER_FILE = 4000
MAX_TOTAL_DIFF_CHARS = 20000
MAX_COMMITS = 20


def github_headers() -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_get(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    response = requests.get(url, headers=github_headers(), params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def github_patch(url: str, body: Dict[str, Any]) -> Any:
    response = requests.patch(url, headers=github_headers(), json=body, timeout=60)
    response.raise_for_status()
    return response.json()


def get_pr() -> Dict[str, Any]:
    url = f"{GITHUB_API_URL}/repos/{REPOSITORY}/pulls/{PR_NUMBER}"
    return github_get(url)


def get_pr_commits() -> List[Dict[str, Any]]:
    url = f"{GITHUB_API_URL}/repos/{REPOSITORY}/pulls/{PR_NUMBER}/commits"
    commits = github_get(url, params={"per_page": MAX_COMMITS})
    return commits[:MAX_COMMITS]


def get_pr_files() -> List[Dict[str, Any]]:
    url = f"{GITHUB_API_URL}/repos/{REPOSITORY}/pulls/{PR_NUMBER}/files"
    files = github_get(url, params={"per_page": 100})
    return files[:MAX_FILES]


def summarize_commits(commits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    summarized = []
    for commit in commits:
        commit_data = commit.get("commit", {})
        message = commit_data.get("message", "").strip()
        first_line = message.splitlines()[0] if message else "No commit message"
        summarized.append(
            {
                "sha": commit.get("sha", "")[:7],
                "author": (commit_data.get("author") or {}).get("name", "unknown"),
                "message": first_line,
            }
        )
    return summarized


def summarize_files(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summarized = []
    total_diff_chars = 0

    for file in files:
        patch = file.get("patch") or ""
        if patch:
            remaining = MAX_TOTAL_DIFF_CHARS - total_diff_chars
            if remaining <= 0:
                patch = "[diff omitted: total diff limit reached]"
            else:
                patch = patch[: min(MAX_PATCH_CHARS_PER_FILE, remaining)]
                total_diff_chars += len(patch)
        else:
            patch = "[binary file or diff unavailable]"

        summarized.append(
            {
                "filename": file.get("filename"),
                "status": file.get("status"),
                "additions": file.get("additions", 0),
                "deletions": file.get("deletions", 0),
                "changes": file.get("changes", 0),
                "patch": patch,
            }
        )

    return summarized


def build_prompt(pr: Dict[str, Any], commits: List[Dict[str, Any]], files: List[Dict[str, Any]]) -> str:
    prompt_payload = {
        "repository": REPOSITORY,
        "pull_request": {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "author": (pr.get("user") or {}).get("login"),
            "base_branch": (pr.get("base") or {}).get("ref"),
            "head_branch": (pr.get("head") or {}).get("ref"),
            "body": pr.get("body") or "",
        },
        "commits": summarize_commits(commits),
        "changed_files": summarize_files(files),
    }

    instructions = textwrap.dedent(
        """
        You are generating a GitHub pull request description.

        Write a clear, concise, professional PR description in GitHub markdown.
        The response must be only the PR description body text.

        Use this structure:
        ## Summary
        - 2 to 5 bullet points

        ## Changes
        - group related changes into bullets

        ## Testing
        - include a short testing note
        - if testing information is not available, say: Not specified.

        ## Notes
        - mention important risks, migrations, follow-ups, or assumptions if they are evident from the diff
        - if none are evident, say: None.

        Requirements:
        - Be faithful to the provided PR data.
        - Do not invent behavior, testing, or business context.
        - Keep it readable and reasonably compact.
        - Mention notable files or areas changed when useful.
        """
    ).strip()

    return f"{instructions}\n\nPR data:\n{json.dumps(prompt_payload, indent=2)}"


def generate_description(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    description = response.output_text.strip()
    if not description:
        raise RuntimeError("OpenAI returned an empty PR description.")
    return description


def update_pr_description(body: str) -> None:
    url = f"{GITHUB_API_URL}/repos/{REPOSITORY}/pulls/{PR_NUMBER}"
    github_patch(url, {"body": body})


def main() -> None:
    pr = get_pr()
    commits = get_pr_commits()
    files = get_pr_files()

    prompt = build_prompt(pr, commits, files)
    description = generate_description(prompt)
    update_pr_description(description)

    print("PR description updated successfully.")
    print("\nGenerated description preview:\n")
    print(description)


if __name__ == "__main__":
    main()
