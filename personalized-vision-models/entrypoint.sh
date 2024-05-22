#!/bin/bash
set -e

sudo service sssd start
sudo service ssh start

# Run the task
pip install transformers
pip install tiktoken
exec "$@"
