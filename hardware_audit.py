from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from whisper_pularr.runtime import runtime_from_hardware_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the read-only remote hardware audit over SSH.")
    parser.add_argument("--host", required=True, help="SSH target such as ubuntu@38.128.232.13")
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity", default=None)
    parser.add_argument("--check-network", action="store_true")
    parser.add_argument("--insecure-host-key-bypass", action="store_true")
    return parser.parse_args()


def build_ssh_command(args: argparse.Namespace) -> list[str]:
    command = ["ssh"]
    if args.identity:
        command.extend(["-i", args.identity])
    if args.insecure_host_key_bypass:
        command.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
    command.append(args.host)
    if args.check_network:
        command.append("AUDIT_NETWORK=1 bash -s")
    else:
        command.append("bash -s")
    return command


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).parent / "remote" / "hardware_audit.sh"
    ssh_command = build_ssh_command(args)
    script_contents = script_path.read_text(encoding="utf-8")
    completed = subprocess.run(
        ssh_command,
        input=script_contents,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip() or "hardware audit failed")

    report = json.loads(completed.stdout)
    recommendation = runtime_from_hardware_report(report).to_dict()
    payload = {
        "hardware_report": report,
        "runtime_recommendation": recommendation,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
