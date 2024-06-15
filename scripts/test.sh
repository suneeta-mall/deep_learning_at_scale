#!/bin/bash

set -eox pipefail

pytest -v  "--cov=prime_radiant --cov-report=html:/tmp --cov-report=term-missing --ignore=tmp/"
