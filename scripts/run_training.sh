#!/bin/bash

if [ "$1" = "price-model" ]; then
    python -m src.training.train_price_model
elif [ "$1" = "vin-model" ]; then
    python -m src.training.train_vin_detector
else
    echo "Usage: "
    echo "  price-model - for training price model"
    echo "  vin-model - for training vin detection model"
    echo "  help - to see this message"
fi