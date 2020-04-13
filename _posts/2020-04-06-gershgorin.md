---
title: "Haben Wir Eine Matrix"
last_modified_at: 2019-06-20T13:51:02-05:00
categories:
  - Blog
tags:
  - linear systems
  - controls
---

# Gershgorin Circle Theorem
## 1. Introduction
Let's talk about linear systems. It's really important to understand the stability of the system. Generally, we teach students to do some form of eigenvalue decomposition and apply the Routh–Hurwitz stability criterion. 

However, these decompositions are notorious for their computational intensity. Enter Prof. Semyon Aranovich Gershgorin from Petrograd Technological Institute. His approach to this problem is so creative that I felt the need to share it. 

![alt text](/assets/images/einen_matrix.png "Original Paper")

## 2. Some Intuition
Take a diagonal matrix. Let's say this one:

![alt text](/assets/images/diagonal.png "Diagonal Matrix")

Obviously, there's no problem here. The eigenvalues of the matrix are simply the diagonal entries (-5, -4, -7). Now we ask a clever question: How different is that diagonal matrix from this one? 

![alt text](/assets/images/less_diagonal.png "Diagonal Matrix")

How much does a slight change in the off diagonal terms change the eigenvalues? Hopefully, your intuition suggests that the they don't change much. The new eigenvalues are -5.0001, -3.9999, and -7. Perhaps you could somehow bound the magnitude of the off-diagonal terms to the change in eigenvalue. Now, your inner linear functional analyst should be screaming: "USE THE NORM"

## 3. Statement of Theorem
Sum the absolute value of the off-diagonal terms in each row. This is your Gershgorin circle radius. The value of the on-diagonal term is your Gershgorin circle center. **Every eigenvalue of the system falls within at least one of the Gershgorin circles** 

The proof for this theorem involves more discussion of norms and eigenvectors and triangle inequalities than I want to deal with, but a good outline is [here](https://mathworld.wolfram.com/GershgorinCircleTheorem.html).

Let's think back to the almost diagonal matrix above. If we sum the off diagonal terms, we see that our Gershgorin circle radius is very small. We know the eigenvalues fall inside that very small circle because the matrix is almost diagonal. It's a neat trick to extend the radius as your matrix becomes less diagonal, while still knowing the eigenvalues fall inside. 

## 4. Application
This theorem is best described and appreciated with a visualization. Here is a simple web app I build to visualize the circles. You will be able to modify the circle radius and circle center for a 4x4 matrix. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tdgriffith/gershgorin/master?urlpath=%2Fvoila%2Frender%2F2020-04-06_Gershgorin-interactive_01.ipynb)

This clever approach of bounding the neighborhood an eigenvalue may fall in is well applied as a first check for any large system matrix. With only basic mathematical operations, we can get some feel for the nature of the system in a very visual manner.







