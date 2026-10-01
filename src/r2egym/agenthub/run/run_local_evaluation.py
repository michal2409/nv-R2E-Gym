import argparse
import json
import pathlib

from r2egym.agenthub.runtime.local import LocalRuntime
from r2egym.commit_models.diff_classes import ParsedCommit


def evaluate(args) -> tuple[dict[str, bool], str | None]:
    """Returns a tuple of the evaluation report and the test output if available."""
    with open(args.dataset, "r") as fin:
        for line in fin:
            row = json.loads(line)
            if row["instance_id"] == args.instance_id:
                dataset_row = row
                break
        else:
            raise ValueError(f"Could not find instance_id {args.instance_id} in dataset {args.dataset}")

    if args.predictions_path == "gold":
        commit = ParsedCommit(**json.loads(dataset_row["parsed_commit_content"]))
        patch = commit.get_patch()
    elif args.predictions_path == "empty":
        patch = None
    else:
        with open(args.predictions_path, "r") as fin:
            for line in fin:
                row = json.loads(line)
                if row["instance_id"] == args.instance_id:
                    patch = row["model_patch"]
                    break
            else:
                raise ValueError(
                    f"Could not find instance_id {args.instance_id} in predictions file {args.predictions_path}"
                )
        if patch is None or not patch.strip():
            print("Empty patch.")
            return {"resolved": False, "patch_exists": False, "patch_successfully_applied": False}, None

    runtime = LocalRuntime(dataset_row)

    if args.predictions_path != "empty":
        # The patch may include untracked R2E-Gym files
        # that are in /testbed when the container starts, but not actually part of the repo (e.g. run_tests.sh).
        # Any changes to these files are ignored using the `--exclude` option of `git apply`.

        print("Getting untracked files from /testbed...")
        git_ls_output, exit_code = runtime.run("git ls-files --others --exclude-standard")
        if exit_code != "0":
            print("Failed to get untracked files from /testbed.")
            return {"resolved": False, "patch_exists": True, "patch_successfully_applied": False}, None

        untracked_files = [file.strip() for file in git_ls_output.split()]
        exclude_str = " ".join(f"--exclude={file}" for file in untracked_files)

        patch_path = f"/tmp/{args.instance_id}.patch"
        pathlib.Path(patch_path).write_text(patch)

        apply_patch_command = f"git apply --whitespace=fix {exclude_str} {patch_path}"
        print(f"Applying patch... Command: {apply_patch_command}")
        _, exit_code = runtime.run(apply_patch_command)
        if exit_code != "0":
            print("Failed to apply patch.")
            return {"resolved": False, "patch_exists": True, "patch_successfully_applied": False}, None

        print("Patch applied successfully. Running evaluation...")

    runtime.setup_env(install_agent_dependencies=False)
    reward, test_output = runtime._calculate_reward(get_test_output=True, timeout=args.timeout)
    resolved = bool(reward)  # reward is always 1 or 0
    return {"resolved": resolved, "patch_exists": True, "patch_successfully_applied": True}, test_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation harness for the given dataset and prediction.")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to a JSON file with the dataset",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--instance_id",
        type=str,
        help="Instance ID to run",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--predictions_path",
        type=str,
        help=(
            "Path to the predictions file in SWE-bench format. "
            "If 'gold', uses the gold patch for this instance. "
            "If 'empty', uses an empty patch."
        ),
        required=True,
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Folder to store output files",
    )

    args = parser.parse_args()

    report, test_output = evaluate(args)

    report_json = json.dumps({args.instance_id: report})
    print(f"Evaluation complete. Report: {report_json}")

    if args.output_dir is None:
        output_dir = pathlib.Path("eval-outputs") / args.instance_id
    else:
        output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.json"
    report_path.write_text(report_json)
    print(f"Report written to {report_path}")

    if test_output is not None:
        test_output_path = output_dir / "test_output.txt"
        test_output_path.write_text(test_output)
        print(f"Test output written to {test_output_path}")
