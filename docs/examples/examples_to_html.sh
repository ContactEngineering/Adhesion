#!/usr/bin/env bash

for f in *.py
do
  jupytext --to notebook --output - $f | jupyter nbconvert --execute --allow-errors -y --stdin --to=html --output=${f%*.py}.html
done