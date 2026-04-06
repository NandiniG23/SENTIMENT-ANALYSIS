# TMDL Guide — History_Open Measures & _Date_tracker

## Overview

This document explains the TMDL model structure used to replace the original `History_Open` calculated table with a **measures-only** approach backed by a dedicated `_Date_tracker` date table.

---

## Problem: Partition Requirement

Every table in the Tabular model engine (Analysis Services / Power BI Premium) **must have at least one partition with `dataView: full`**.  
A table that contains only measures and has no partition will cause a load error:

```
The table 'History_Open' does not have any partitions of type Full.
```

### Solution Applied

A dummy calculated partition has been added to `History_Open`:

```tmdl
partition History_Open = calculated
    mode: import
    dataView: full
    source = ```
            DATATABLE ( "__placeholder__", INTEGER, {} )
            ```
```

This partition produces **zero rows**. It exists solely to satisfy the engine requirement.  
All business logic lives in the measures; no column from this partition is exposed or used in visuals.

---

## Files

| File | Purpose |
|------|---------|
| `PowerBI/tables/History_Open.tmdl` | All measures + dummy partition |
| `PowerBI/tables/_Date_tracker.tmdl` | Calendar/date dimension table |

---

## _Date_tracker Table

The `_Date_tracker` table replaces the old `History_Open` calculated table as the source of dates. It is defined as:

```dax
ADDCOLUMNS (
    CALENDAR ( DATE ( 2024, 1, 1 ), TODAY () - 1 ),
    "DateKey",     YEAR ( [Date] ) * 10000 + MONTH ( [Date] ) * 100 + DAY ( [Date] ),
    "Date_Text",   FORMAT ( [Date], "d-MMM-yy" ),
    "Month MM-YY", FORMAT ( [Date], "MMM-yy" ),
    "Month Index", YEAR ( [Date] ) * 100 + MONTH ( [Date] )
)
```

Its partition already carries `dataView: full`, so no dummy partition is needed there.

### Required Relationships

Connect `_Date_tracker[Date]` to the fact tables:

| From | To | Active? |
|------|----|---------|
| `_Date_tracker[Date]` | `Open_Order[Created Date]` | Yes |
| `_Date_tracker[Date]` | `Open_Order[Effective Billing Date]` | No — use `USERELATIONSHIP()` in `Move to Billing` if needed |
| `_Date_tracker[Date]` | `Closed_order[Created_Date_Only]` | No — filter directly in measure |
| `_Date_tracker[Date]` | `Closed_order[Closed_Date_Only]` | No — filter directly in measure |

---

## Applying Changes in Tabular Editor

1. Open **Tabular Editor 2** or **Tabular Editor 3**.
2. Connect to your Power BI Desktop model via the external tools ribbon, or open the `.pbip` / `.bim` file.
3. From the **File** menu choose **Open → From Folder (TMDL)** and point to the `PowerBI/` directory.
4. Review the model tree — `History_Open` and `_Date_tracker` should appear.
5. Click **Model → Deploy** (or press **F5**) to apply changes.
6. Verify there are no partition errors in the **Messages** pane.

### Applying via Power BI Desktop (manual paste)

1. Open the **DAX editor** in a blank Power BI Desktop file.
2. From the **Modeling** ribbon, use **New Table** and paste the `_Date_tracker` DAX expression.
3. Create each measure from the **Modeling → New Measure** dialog, copying the DAX from `History_Open.tmdl`.
4. The dummy `DATATABLE` partition is **not needed** when working directly in Power BI Desktop — Power BI Desktop automatically manages partitions for calculated tables and measure groups.

> **Note**: The dummy partition is only required when you deploy the model from TMDL directly to Analysis Services or Power BI Premium via Tabular Editor / XMLA endpoint.

---

## TMDL Comment Syntax

TMDL uses `///` for documentation comments (triple-slash), **not** `//` (double-slash).  
Using `//` inside a TMDL document causes an **indentation/parsing error**.

Correct usage:
```tmdl
/// This is a valid TMDL comment.
table MyTable
    ...
```

Incorrect — do not use:
```tmdl
// This will cause a parse error in TMDL.
table MyTable
    ...
```

---

## Measures Reference

All measures live in the `History_Open` table and reference `_Date_tracker[Date]` for date context.

| Measure | Description |
|---------|-------------|
| `Created(Open File)` | Count of Open_Order rows created on the selected date with status Manage/Release |
| `created (Closed File)` | Distinct count of Closed_order rows created on the selected date |
| `Total New Created(Open+Closed)` | Sum of the two creation measures |
| `Acutal closed` | Distinct count of orders closed on the selected date (name kept as-is from original model) |
| `Move to Billing` | Count of orders moved to billing on the selected date |
| `Total Closed(Closed+billing)` | Currently equals `[Acutal closed]` — change to `[Acutal closed] + [Move to Billing]` if required |
| `Running Open` | Cumulative new created up to the selected date |
| `Running Total Closed (Closed+billing)` | Cumulative closed up to the selected date |
| `Acutal open` | `Running Open` − `Running Total Closed` (name kept as-is from original model) |
| `OneOff_Plus_Monthly_EUR_Closed_Factored` | Daily closed order amount (₹) |
| `OneOff_Chgs_EUR_Open_Factored` | Daily open order amount × 100.351 (₹) |
| `Total Closed Amt` | Total of `Closed_order[One Off Total Gold]` in filter context |
| `Total New Created (All)` | Alias for `Total New Created(Open+Closed)` |
| `Current Month Total New Created` | MTD new created |
| `MTD_Closed_Amount` | MTD closed amount |
| `MTD_Open_Amount` | MTD open amount |
| `Total in billing` | Alias for `Move to Billing` |
| `Feb_26 Closed Amt` | Total Closed Amt filtered to Feb-26 |
| `Mar_26 Closed Amt` | Total Closed Amt filtered to Mar-26 |
