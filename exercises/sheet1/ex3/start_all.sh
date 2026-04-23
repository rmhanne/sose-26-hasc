#!/bin/sh


set -e
found=0

for file in ./compiler_flags_*; do
  [ -e "$file" ] || continue

  if [ -f "$file" ] && [ -x "$file" ]; then
    found=1
    echo "Running $file ..."
    "$file"
    echo "Finished $file"
    echo
  fi
done

if [ "$found" -eq 0 ]; then
  echo "No executable files matching compiler_flags_* found. Make sure you run make first."
fi