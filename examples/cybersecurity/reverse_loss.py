import numpy as np

unity_utilisation = 1105
utilisation_coeff = 0.2
accuracy_coeff = 0.8
target_accuracy = 0.92

def compute_utilisation(layers):

        def lut_cost(X, Y): 
            return round((float(Y) / 3) * (np.power(2.0, (float(X) - 4)) - np.power(-1.0, float(X))))

        util = 0
        for layer in layers:
            util += layer * lut_cost(1, 2)

        return util

def reverse_loss(layers, loss):

    utilisation_loss = (compute_utilisation(layers) / unity_utilisation)
    accuracy_loss = (loss - utilisation_loss * utilisation_coeff) / accuracy_coeff

    return (target_accuracy - accuracy_loss)


for arch in [[33, 16, 6, 5], [33, 14, 13, 10], [33, 14, 9, 6]]:
    print(f"Accuracy of {arch}: {round(reverse_loss(arch, 0.076), 3)}, Utilisation: {compute_utilisation(arch)}")

nid_s_arch = [593, 100]

print(f"Accuracy of {arch}: {84}, Utilisation: {compute_utilisation(nid_s_arch)}")