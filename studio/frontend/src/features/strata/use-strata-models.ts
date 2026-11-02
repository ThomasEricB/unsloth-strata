// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { type StrataModel, listStrataModels } from "./api";

// Shared across pickers; refreshed each time a picker opens, since Strata can set up a model meanwhile.
let cached: StrataModel[] | null = null;

/** The local Strata install's models, or null while unknown, when Strata is absent, or for
 *  accounts that may not load them (the route answers 403). */
export function useStrataModels(active = true): StrataModel[] | null {
  const [models, setModels] = useState<StrataModel[] | null>(cached);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    listStrataModels(controller.signal)
      .then((next) => {
        cached = next;
        setModels(next);
      })
      .catch(() => {
        if (!controller.signal.aborted) setModels(null);
      });
    return () => controller.abort();
  }, [active]);
  return models;
}
