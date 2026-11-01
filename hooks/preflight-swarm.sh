#!/bin/bash
# preflight-swarm.sh — retired.
#
# This hook used to spawn a Lateral Thinker subagent and inject a
# <subconscious_preflight> block that claimed a Haiku model had already run
# and told the main agent not to mention it. That block is not a memory,
# it is not a recorded edge, and it is not injected.
#
# The file remains so an older settings.json that still points at it fails
# open instead of erroring. It never calls a model.

exit 0
