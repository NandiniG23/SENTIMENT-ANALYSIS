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


## Power BI Model – History_Open Measures Table

### File: `History_Open.tmdl`

`History_Open` is a **measures-only table** that holds all operational KPI measures for the order-tracking dashboard. It does **not** store any rows; all date context is supplied at report runtime by the `_Date_tracker` calendar table.

#### Why the dummy partition is needed
Power BI's Tabular engine requires every table to have at least one partition in the **Full DataView** before it can validate or deploy the model. A measures-only table has no natural data source, so a one-row calculated partition is added:

```dax
ROW ( "_dummy", 0 )
```

The `dataView: full` property on this partition satisfies the engine requirement without adding any meaningful data.

#### Measures included
| Measure | Description |
|---|---|
| `Created(Open File)` | Daily count of new open orders |
| `created (Closed File)` | Daily count of newly created closed orders |
| `Total New Created(Open+Closed)` | Sum of the two created counts |
| `Acutal closed` | Daily count of orders closed on that date |
| `Move to Billing` | Daily count of orders moved to billing |
| `Total Closed(Closed+billing)` | Alias for `Acutal closed` |
| `Running Open` | Cumulative new-created total up to the selected date |
| `Running Total Closed (Closed+billing)` | Cumulative closed total up to the selected date |
| `Acutal open` | Running Open minus Running Total Closed |
| `OneOff_Plus_Monthly_EUR_Closed_Factored` | Daily closed EUR amount from `Closed_order` |
| `OneOff_Chgs_EUR_Open_Factored` | Daily open EUR amount from `Open_Order` (×100.351) |
| `Total Closed Amt` | Total closed amount across all dates |
| `Total New Created (All)` | Alias for `Total New Created(Open+Closed)` |
| `Current Month Total New Created` | MTD count of new-created orders |
| `MTD_Closed_Amount` | Month-to-date closed EUR amount |
| `MTD_Open_Amount` | Month-to-date open EUR amount |
| `Total in billing` | Alias for `Move to Billing` |
| `Feb_26 Closed Amt` | `Total Closed Amt` filtered to Feb-26 |
| `Mar_26 Closed Amt` | `Total Closed Amt` filtered to Mar-26 |

#### How to deploy
1. Place `History_Open.tmdl` in the TMDL model folder alongside other table files (e.g. `_Date_tracker.tmdl`).
2. Open the model folder in **Tabular Editor 3** (or use the Power BI TMDL deployment API).
3. Validate the model — the dummy partition ensures no *"Every table must contain at least one partition in the Full DataView"* error is raised.
4. Deploy to your Power BI Premium/Fabric workspace.
5. In report visuals, use `_Date_tracker[Date]` (or `_Date_tracker[Month MM-YY]`) on the axis and the measures from `History_Open` in the Values well.

#### Relationship requirements
Ensure `_Date_tracker[Date]` is related to:
- `Open_Order[Created Date]` (active or via `USERELATIONSHIP`)
- `Open_Order[Effective Billing Date]`
- `Closed_order[Created_Date_Only]`
- `Closed_order[Closed_Date_Only]`

---

## OutPut
#### POSITIVE:
![image](https://github.com/user-attachments/assets/6f27ae3d-54c4-47c6-b3a6-42560e579ffd)

#### NEGATIVE:
![image](https://github.com/user-attachments/assets/25a9a2d9-d034-4b3d-bb63-5081f43dd23a)

