"""Script to run test suite."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def run_pytest(
    test_dir: str = "tests",
    verbose: bool = True,
    coverage: bool = False,
    markers: str = None,
    failfast: bool = False
) -> int:
    """
    Run pytest programmatically.

    Args:
        test_dir: Directory containing tests
        verbose: Verbose output
        coverage: Generate coverage report
        markers: Pytest markers to filter tests
        failfast: Stop on first failure

    Returns:
        Exit code
    """
    cmd = [sys.executable, "-m", "pytest", test_dir]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    if markers:
        cmd.extend(["-m", markers])

    if failfast:
        cmd.append("-x")

    cmd.append("--tb=short")

    logger.info(f"Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)

    return result.returncode


def run_unit_tests(failfast: bool = False) -> int:
    logger.info("Running unit tests...")
    return run_pytest(test_dir="tests", markers="not integration", failfast=failfast)


def run_integration_tests(failfast: bool = False) -> int:
    logger.info("Running integration tests...")
    return run_pytest(test_dir="tests", markers="integration", failfast=failfast)


def run_all_tests(coverage: bool = False, failfast: bool = False) -> int:
    logger.info("Running all tests...")
    return run_pytest(test_dir="tests", coverage=coverage, failfast=failfast)


def run_specific_test(test_file: str, failfast: bool = False) -> int:
    logger.info(f"Running specific test: {test_file}")
    return run_pytest(test_dir=test_file, verbose=True, failfast=failfast)


def main():
    parser = argparse.ArgumentParser(description="Run test suite")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")

    args = parser.parse_args()

    setup_logging()

    logger.info("=" * 60)
    logger.info("RUNNING TEST SUITE")
    logger.info("=" * 60)

    if args.file:
        exit_code = run_specific_test(args.file, failfast=args.failfast)
    elif args.unit:
        exit_code = run_unit_tests(failfast=args.failfast)
    elif args.integration:
        exit_code = run_integration_tests(failfast=args.failfast)
    else:
        exit_code = run_all_tests(coverage=args.coverage, failfast=args.failfast)

    logger.info("=" * 60)
    if exit_code == 0:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error(f"TESTS FAILED (exit code: {exit_code})")
    logger.info("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
