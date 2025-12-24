#!/bin/bash
# Wrapper script for match_maps that sets LD_LIBRARY_PATH
# This ensures libpmc.so and other /usr/local/lib libraries are found

export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
exec "$(dirname "$0")/build/bin/match_maps" "$@"




