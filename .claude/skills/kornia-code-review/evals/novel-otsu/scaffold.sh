#!/usr/bin/env bash
exec bash "$(dirname "$(realpath "$0")")/../scaffold.sh" "$(dirname "$(realpath "$0")")"
