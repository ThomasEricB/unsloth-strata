# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Strata models for the picker: the run configs of the local Strata install, and the state of
Studio's own Strata server. Loading and unloading go through /api/inference/load and /unload
with a ``strata:<model_name>`` model path."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from auth import policy
from auth.authentication import get_current_subject
from core.inference.strata_backend import get_strata_backend

router = APIRouter()


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    # Studio starts Strata as its own user, with that user's models and memory.
    await policy.require_owner()


@router.get("/status", dependencies = [Depends(_require_installation_owner)])
async def strata_status():
    """Whether Strata is installed, what Studio's Strata server is serving, and other engines."""
    return await asyncio.to_thread(get_strata_backend().status)


@router.get("/models", dependencies = [Depends(_require_installation_owner)])
async def list_strata_models():
    models = await asyncio.to_thread(get_strata_backend().catalog)
    return {"models": [model.to_json() for model in models]}
