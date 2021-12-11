# Spiking Neural Network for SAR data

SNN designed with SpykeTorch for classifying Synthetic Aperture Radar imagery data. The SNN uses reinforcement and STDP based learning with lateral inhibition to train a network of neurons.

## Usage

Running the following will train the network on the MSTAR Synthetic Aperture Radar dataset. It uses the SN_9563 and SN_C71 vehicles and can classify with ~90% percent accuracy on the test set.

```bash
python MStar.py
```

## References
https://github.com/miladmozafari/SpykeTorch