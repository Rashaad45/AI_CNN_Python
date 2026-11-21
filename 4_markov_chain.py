"""
PROJECT 4: MARKOV CHAIN TEXT GENERATOR
--------------------------------------
Markov assumption: the next state depends ONLY on the current state.
    P(x_{t+1} | x_t, x_{t-1}, ..., x_0) = P(x_{t+1} | x_t)

Part A: Weather chain (HOT / COLD / WARM) with a transition matrix P
        - simulate the chain
        - find the stationary distribution  pi = pi P
Part B: Text generator
        - count which word follows which
        - turn counts into probabilities
        - walk the chain to write new text

Run:  python 4_markov_chain.py
Needs: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(7)

# =================================================
# PART A: WEATHER MARKOV CHAIN
# =================================================
states = ["HOT", "COLD", "WARM"]

# P[i][j] = probability of going from state i to state j
# Each ROW must add up to 1.
P = np.array([
    #  HOT  COLD  WARM
    [0.5, 0.3, 0.2],   # from HOT
    [0.2, 0.6, 0.2],   # from COLD
    [0.2, 0.3, 0.5],   # from WARM
])
print("Transition matrix P:")
print(P)
print("Row sums:", P.sum(axis=1), "(must all be 1)")

# ---- Simulate 10,000 days ----
current = 0                      # start on a HOT day
days = 10000
visits = np.zeros(3)
first_days = []

for day in range(days):
    visits[current] += 1
    if day < 15:
        first_days.append(states[current])
    current = np.random.choice(3, p=P[current])   # pick tomorrow using row of P

print("\nFirst 15 simulated days:", " -> ".join(first_days))
simulated_share = visits / days

# ---- Stationary distribution, method 1: keep multiplying pi_{t+1} = pi_t P ----
pi = np.array([1.0, 0.0, 0.0])   # start 100% HOT
history = [pi.copy()]
for step in range(30):
    pi = pi @ P
    history.append(pi.copy())
history = np.array(history)

# ---- Stationary distribution, method 2: eigenvector of P^T with eigenvalue 1 ----
eigenvalues, eigenvectors = np.linalg.eig(P.T)
index = np.argmin(np.abs(eigenvalues - 1))
pi_eig = np.real(eigenvectors[:, index])
pi_eig = pi_eig / pi_eig.sum()   # scale so it adds up to 1

print("\nLong-run share of days:")
print(f"{'state':6s} {'simulated':>10s} {'power method':>13s} {'eigenvector':>12s}")
for i in range(3):
    print(f"{states[i]:6s} {simulated_share[i]:10.3f} {pi[i]:13.3f} {pi_eig[i]:12.3f}")

# Probability of weather 2 days from now, starting HOT: row 0 of P^2
P2 = np.linalg.matrix_power(P, 2)
print("\nIf today is HOT, chances in 2 days:",
      {states[j]: round(float(P2[0, j]), 3) for j in range(3)})

# =================================================
# PART B: MARKOV CHAIN TEXT GENERATOR
# =================================================
text = """
the engine turns the shaft and the shaft turns the wheel and the wheel moves the car
the driver turns the wheel and the car turns left and the road turns right
the team builds the car and the car races on the track and the track is rough
the car needs a strong frame and the frame holds the engine and the engine needs fuel
the team tests the car on the track and the driver pushes the car to the limit
a good design makes the car light and a light car moves fast on the rough track
"""


def build_markov_model(text, n=1):
    """
    n = how many previous words make up the 'state'.
    n=1 -> next word depends on 1 word, n=2 -> on the last 2 words.
    """
    words = text.lower().split()
    counts = defaultdict(lambda: defaultdict(int))

    # Count: after this state, which word came next?
    for i in range(len(words) - n):
        state = tuple(words[i:i + n])
        next_word = words[i + n]
        counts[state][next_word] += 1

    # Turn counts into probabilities
    model = {}
    for state, next_words in counts.items():
        total = sum(next_words.values())
        model[state] = {}
        for word, count in next_words.items():
            model[state][word] = count / total
    return model


def generate_text(model, start, length=30):
    state = tuple(start)
    result = list(state)
    n = len(state)

    for _ in range(length):
        options = model.get(state)
        if options is None:            # dead end: no word ever followed this state
            break
        words = list(options.keys())
        probs = list(options.values())
        next_word = np.random.choice(words, p=probs)
        result.append(next_word)
        state = tuple(result[-n:])     # slide the window forward
    return " ".join(result)


model1 = build_markov_model(text, n=1)
model2 = build_markov_model(text, n=2)

print("\nWhat follows the word 'the'? (n=1 model)")
for word, p in sorted(model1[("the",)].items(), key=lambda kv: -kv[1]):
    print(f"  {word:8s} {p:.2f}")

print("\nGenerated text, n=1 (more random):")
print(" ", generate_text(model1, ["the"], length=25))

print("\nGenerated text, n=2 (more like the original):")
print(" ", generate_text(model2, ["the", "car"], length=25))

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i in range(3):
    axes[0].plot(history[:, i], marker="o", markersize=3, label=states[i])
axes[0].set_xlabel("Step t")
axes[0].set_ylabel("Probability")
axes[0].set_title("pi_(t+1) = pi_t P  converges to the stationary distribution")
axes[0].legend()

im = axes[1].imshow(P, cmap="Blues", vmin=0, vmax=1)
axes[1].set_xticks(range(3))
axes[1].set_yticks(range(3))
axes[1].set_xticklabels(states)
axes[1].set_yticklabels(states)
axes[1].set_xlabel("to state")
axes[1].set_ylabel("from state")
axes[1].set_title("Transition matrix P")
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, P[i, j], ha="center", va="center")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("4_markov_chain.png", dpi=120)
print("\nPlot saved as 4_markov_chain.png")
plt.show()
