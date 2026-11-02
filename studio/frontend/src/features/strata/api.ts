// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** Keep in sync with strata_backend.py's MODEL_PREFIX. */
export const STRATA_MODEL_PREFIX = "strata:";

export function isStrataModelId(
  value: string | null | undefined,
): value is string {
  return typeof value === "string" && value.startsWith(STRATA_MODEL_PREFIX);
}

/** One run config of the local Strata install, served with Strata's own settings. */
export interface StrataModel {
  id: string;
  model_path: string;
  config_path: string;
  supports_vision: boolean;
  supports_reasoning: boolean;
  supports_tools: boolean;
  max_context_length: number | null;
  resident_budget_gib: number | null;
}

export function strataRowsFor(
  models: StrataModel[] | null,
  query: string,
): StrataModel[] {
  const needle = query.trim().toLowerCase();
  return (models ?? []).filter(
    (model) => !needle || model.id.toLowerCase().includes(needle),
  );
}

export function strataContextLabel(tokens: number | null): string | null {
  if (!tokens) return null;
  return tokens >= 1024
    ? `${Math.round(tokens / 1024)}K context`
    : `${tokens} context`;
}

export async function listStrataModels(
  signal?: AbortSignal,
): Promise<StrataModel[]> {
  const response = await authFetch("/api/strata/models", { signal });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Could not list Strata models"),
    );
  }
  return ((await response.json()) as { models: StrataModel[] }).models;
}
