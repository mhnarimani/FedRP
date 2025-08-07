# FedRP: A Communication-Efficient Approach for Differentially Private Federated Learning Using Random Projection

This repository contains the implementation of **FedRP**, a novel federated learning algorithm that enhances communication efficiency and privacy by combining **Random Projection** with the **ADMM optimization framework**.

FedRP is designed for federated learning settings and provides a strong (ϵ, δ)-Differential Privacy guarantee without sacrificing model accuracy.

---

## 📌 Paper

> **FedRP: A Communication-Efficient Approach for Differentially Private Federated Learning Using Random Projection**  
> [Author(s): YOUR NAME(S)]  
> Preprint submitted to *Expert Systems With Applications*, July 30, 2025.

[📄 PDF of the Paper](./Titlelabel1\ (8).pdf)

---

## 🧠 Method Overview

FedRP introduces a secure dimension-reduction step (via random projection) on the client side before model updates are sent to the server. This significantly reduces communication costs and enhances privacy by preventing reconstruction attacks.

- Clients use a **shared random projection matrix** to project model parameters.
- Optimization is performed using **Consensus ADMM** in the projected space.
- FedRP avoids the need to reverse the projections, unlike FedSketch and similar methods.

![Architecture](docs/architecture.png) <!-- Optional: Add illustration -->

---

## 📂 Directory Structure

