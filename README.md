# SENTIMENT-ANALYSIS

COMPANY : CODTECH IT SOLUTIONS

NAME : Nandani Gulab Gupta

INTERN ID : CT08DK744

Domain : Data Analytics

Duration : 8 WEEKS

MENTOR : Neela Santosh

## Description
This repositry focuses on sentiment analysis using natural language processing(nlp).For this project was built using jupyter notebook in vs code and showcases the application of nlp techniques for real-world text classification.The objective was to classify reviews as positive or negative based on the content.

## Tools & Libraries used
Python

Jupyter notebook(vs code)

pandas

numpy

matplotlib

seaborn

nltk

scikit-learn


## OutPut
#### POSITIVE:
![image](https://github.com/user-attachments/assets/6f27ae3d-54c4-47c6-b3a6-42560e579ffd)

#### NEGATIVE:
![image](https://github.com/user-attachments/assets/25a9a2d9-d034-4b3d-bb63-5081f43dd23a)

---

## Power BI Model – History_Measures (TMDL)

The file [`model/History_Measures.tmdl`](model/History_Measures.tmdl) replaces the `History_Open` calculated table with DAX measures that reference the existing `_Date_tracker` calculated date table.

### Prerequisites

| Requirement | Detail |
|---|---|
| Date table | `_Date_tracker` (CALENDAR 2024-01-01 → TODAY()-1, with `Date`, `DateKey`, `Date_Text`, `Month MM-YY`, `Month Index` columns) |
| Data tables | `Open_Order`, `Closed_order` |
| Relationships | `_Date_tracker[Date]` → `Open_Order[Created Date]` (active); additional inactive relationships to `Open_Order[Effective Billing Date]`, `Closed_order[Created_Date_Only]`, `Closed_order[Closed_Date_Only]` |

### Measures

| Measure | Description |
|---|---|
| `Created (Open File)` | Daily count of new open orders for tracked coordinators (Status = Manage/Release) |
| `Created (Closed File)` | Daily count of new closed orders (by created date) |
| `Total New Created (Open+Closed)` | Sum of the two created measures |
| `Actual Closed` | Daily count of orders closed (by closed date) |
| `Move to Billing` | Daily count of orders that moved to billing status |
| `Total Closed (Closed+Billing)` | Currently equals Actual Closed; add `[Move to Billing]` if Closed+Billing is desired |
| `Running Open` | Cumulative new-created total up to selected date (0 for future dates) |
| `Running Total Closed (Closed+Billing)` | Cumulative closed total up to selected date (0 for future dates) |
| `Actual Open` | `Running Open` − `Running Total Closed` |
| `OneOff_Plus_Monthly_EUR_Closed_Factored` | Daily closed order gold amount (One Off Total Gold) |
| `OneOff_Chgs_EUR_Open_Factored` | Daily open order EUR charges × 100.351 |
| `Total Closed Amt` | Total closed amount across all dates |
| `Total New Created (All)` | Alias for `Total New Created (Open+Closed)` |
| `Current Month Total New Created` | MTD total new created |
| `MTD_Closed_Amount` | Month-to-date closed amount |
| `MTD_Open_Amount` | Month-to-date open amount |
| `Feb_26 Closed Amt` | Closed amount for February 2026 |
| `Mar_26 Closed Amt` | Closed amount for March 2026 |
| `Total in billing` | Alias for `Move to Billing` |

### Usage in visuals

1. Remove `History_Open` from your Power BI model.
2. Add the measures to your model using one of these methods:
   - **Tabular Editor 3 (recommended):** Open your model → File → Import → From file → select `model/History_Measures.tmdl`, or paste the file contents into the TMDL editor.
   - **Tabular Editor 2:** Use the Advanced Scripting pane and execute the TMDL via the command-line `te` tool.
   - **TMDL-based project:** If your Power BI project uses a TMDL folder structure (e.g., `.pbip` format), place `History_Measures.tmdl` inside the `definition/tables/` folder.
3. Place `_Date_tracker[Date]` (or `_Date_tracker[Month MM-YY]`) on the visual axis.
4. Use the measures above in the **Values** well — no calculated columns needed.

### DAX patterns used

- **Daily measures** use `VAR d = SELECTEDVALUE(_Date_tracker[Date])` so they respond to a single-date row context.
- **Running totals** use `VAR d = MAX(_Date_tracker[Date])` with `FILTER(ALL(_Date_tracker[Date]), _Date_tracker[Date] <= d)` to accumulate across dates.
- All running totals return `0` for dates beyond `TODAY()`.

