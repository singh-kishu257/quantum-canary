"""Credentials for IBM and IonQ, resolved the way the vendors' own libraries do.

Nothing here searches for files. By default the vendor libraries find the secret
themselves, in their usual order:

  IBM   QiskitRuntimeService():  the account saved with save_account() in
        ~/.qiskit/qiskit-ibm.json, or the QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE
        environment variables.
  IonQ  IonQProvider():  the QISKIT_IONQ_API_TOKEN, IONQ_API_KEY or IONQ_API_TOKEN
        environment variable (qiskit-ionq also reads a .env file).

To use a secret kept in a file, name the file. It is read once, passed straight to the
vendor's constructor, and never printed, logged, saved in a Report or exported to the
environment. Error messages name the file, never its contents.

    service = credentials.ibm_service("ibm_api_token.txt", "ibm_crn_instance.txt")
    provider = credentials.ionq_provider("ionq_api_token.txt")
"""
from __future__ import annotations

import pathlib
from typing import Optional, Union

__all__ = ["read_secret", "ibm_service", "ionq_provider"]

Path = Union[str, pathlib.Path]


def read_secret(path: Path) -> str:
    """The single value in a one-line secret file, whitespace and BOM stripped.
    Raises ValueError naming the file (never its contents) if it cannot be read,
    is empty, or holds more than one value."""
    p = pathlib.Path(path).expanduser()
    try:
        value = p.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise ValueError(f"cannot read {p} ({type(exc).__name__})") from None
    if not value:
        raise ValueError(f"{p} is empty")
    if len(value.split()) != 1:
        raise ValueError(f"{p} should hold a single value on one line")
    return value


def ibm_service(token_file: Optional[Path] = None, instance_file: Optional[Path] = None,
                channel: str = "ibm_quantum_platform", instance: Optional[str] = None):
    """A QiskitRuntimeService. With no files: the saved account or QISKIT_IBM_*
    variables. With token_file: that token, on `channel`. With instance_file (or
    `instance`, given inline): that instance, the name or CRN of your IBM plan,
    instead of the account's default."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = {}
    if token_file is not None:
        kwargs.update(channel=channel, token=read_secret(token_file))
    if instance_file is not None:
        kwargs["instance"] = read_secret(instance_file)
    elif instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def ionq_provider(token_file: Optional[Path] = None):
    """An IonQProvider. With no file: the key from the environment, as qiskit-ionq
    resolves it. Raises ValueError if no key is found at all."""
    from qiskit_ionq import IonQProvider
    provider = IonQProvider(read_secret(token_file) if token_file is not None else None)
    if not provider.credentials.get("token"):
        raise ValueError("no IonQ API key found: pass a token file, or set IONQ_API_KEY")
    return provider
