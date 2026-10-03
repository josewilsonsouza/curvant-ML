#!/usr/bin/env bash
# Execução completa: amostras, modelos tabulares, LSTM e relatórios.
set -e
cd "$(dirname "$0")"
python -W ignore -m risco amostras
python -W ignore -m risco avaliar --modelos RF XGB LGB
python -W ignore -m risco relatorio
python -W ignore -m risco avaliar --modelos LSTM
python -W ignore -m risco relatorio
