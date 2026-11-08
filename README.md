# Machine Learning Bootcamp

A six-day course for mixed beginner engineers and business learners. You work on your own laptop. Nothing in this repository calls a cloud service.

This repository currently contains Day 1. Later days are not here yet.

The class is taught in Thai. These notes are in English.

## Syllabus

### [Day 1: Supervised learning](01-supervised-learning/)

Build two local checks for KlongCart, a fictional Bangkok online grocery. Classification asks who will place no paid order next month. Regression asks how many baht they will spend next month.

In this day, you will:

- Set up Python 3.15.2 with mise and uv.
- Read one synthetic customer table and keep both July outcomes out of the features.
- Fit a churn classifier, compare it with a majority baseline, and choose a voucher threshold on validation.
- Fit a spend regressor on the same rows, compare it with a mean baseline, and read MAE and RMSE in THB.
- Type one customer into a form and read a churn probability, a yes or no, and an estimated spend.

Outcome: a classroom form. The scores are synthetic. They are not a production retention system.

Setup, the Bangkok timetable, the notebooks, the data, and the form are in that folder. Start with [01-supervised-learning/README.md](01-supervised-learning/README.md).
