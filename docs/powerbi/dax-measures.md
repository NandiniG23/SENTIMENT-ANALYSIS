# Power BI DAX Measures — Replacing the `History_Open` Calculated Table

## Overview

The original Power BI model used a **calculated table** called `History_Open` (created with `CALENDAR(DATE(2024,1,1), TODAY()-1)`) that hosted multiple calculated columns for daily counts, amounts, and running totals. Calculated tables with row-by-row calculated columns are refreshed on every model refresh and can significantly increase model size and refresh time.

This document provides **equivalent DAX measures** that replace every calculated column in `History_Open`. The measures work against a proper **Date dimension table** (`DimDate`) and are evaluated dynamically in visual context — no calculated table is needed.

---

## Why Remove the Calculated Table?

| Calculated Table Approach | Measures-Only Approach |
|---|---|
| Rows materialised at refresh | Computed on demand in visual context |
| Grows every day (one row per day) | No stored rows |
| Calculated columns evaluated row-by-row | CALCULATE + FILTER patterns — same logic, no storage |
| Running totals require `ALL(History_Open)` cross-filter | Use `ALL('DimDate')` or `DATESYTD` / `DATESBETWEEN` |
| MTD measures reference `History_Open[Date]` column | MTD measures use `'DimDate'[Date]` via time-intelligence |

---

## Step 1 — Create a Date Dimension Table (`DimDate`)

If your model does not already have a proper Date table, add one. The simplest way is to use a **DAX calculated table** (one-time scaffold, no calculated columns needed beyond the date attributes):

```dax
DimDate =
VAR StartDate = DATE(2024, 1, 1)
VAR EndDate   = TODAY() + 365          -- extend into the future for slicers
RETURN
ADDCOLUMNS(
    CALENDAR(StartDate, EndDate),
    "Year YYYY",        FORMAT([Date], "YYYY"),
    "Month Index",      YEAR([Date]) * 100 + MONTH([Date]),
    "Month MM-YY",      FORMAT([Date], "MMM-yy"),
    "Date Text",        FORMAT([Date], "d-MMM-yy"),
    "Day of Week",      WEEKDAY([Date], 2),
    "Quarter",          "Q" & QUARTER([Date])
)
```

> **Important**: After adding `DimDate`, mark it as the **Date Table** in Power BI Desktop:  
> *Table tools → Mark as date table → Date column = `DimDate[Date]`*

---

## Step 2 — Relationships

Create the following relationships (all **Many-to-One** from the fact table to `DimDate`):

| From Table | From Column | To Table | To Column | Active? |
|---|---|---|---|---|
| `Open_Order` | `Created Date` | `DimDate` | `Date` | ✅ Active |
| `Open_Order` | `Effective Billing Date` | `DimDate` | `Date` | ❌ Inactive (use `USERELATIONSHIP`) |
| `Closed_order` | `Created_Date_Only` | `DimDate` | `Date` | ❌ Inactive (use `USERELATIONSHIP`) |
| `Closed_order` | `Closed_Date_Only` | `DimDate` | `Date` | ❌ Inactive (use `USERELATIONSHIP`) |

> **Note**: Only one relationship to `DimDate` can be active per fact table. Choose the one most commonly used on the visual axis (e.g., `Created Date` for `Open_Order`). The inactive relationships are activated inside specific measures using `USERELATIONSHIP`.

---

## Step 3 — DAX Measures

Place all measures in a dedicated **Measures table** (or in your existing fact tables). The recommended home table is a blank/disconnected helper table called `_Measures`.

---

### 3.1 `Created (Open File)`

Counts open orders created on the selected date.

```dax
Created (Open File) =
VAR CurrentDate = SELECTEDVALUE('DimDate'[Date])
RETURN
IF(
    ISBLANK(CurrentDate),
    BLANK(),
    CALCULATE(
        COUNTROWS(Open_Order),
        FILTER(
            Open_Order,
            DATEVALUE(Open_Order[Created Date]) = CurrentDate
                && TRIM(Open_Order[Status]) IN {"Manage", "Release"}
                && TRIM(Open_Order[Delivery Order Owner])
                    IN {"Amit VETHEKAR", "Rohan Lokhande", "Akshay CHAPKE", "vaibhav rane"}
        )
    )
)
```

---

### 3.2 `Created (Closed File)`

Counts closed orders by their creation date.

```dax
Created (Closed File) =
VAR CurrentDate = SELECTEDVALUE('DimDate'[Date])
RETURN
IF(
    ISBLANK(CurrentDate),
    BLANK(),
    CALCULATE(
        DISTINCTCOUNT(Closed_order[Order Reference]),
        USERELATIONSHIP(Closed_order[Created_Date_Only], 'DimDate'[Date]),
        'DimDate'[Date] = CurrentDate
    )
)
```

---

### 3.3 `Total New Created (Open+Closed)`

Sum of the two creation counts.

```dax
Total New Created (Open+Closed) =
[Created (Open File)] + [Created (Closed File)]
```

---

### 3.4 `Actual Closed`

Distinct closed orders by their closed date.

```dax
Actual Closed =
VAR CurrentDate = SELECTEDVALUE('DimDate'[Date])
RETURN
IF(
    ISBLANK(CurrentDate),
    BLANK(),
    CALCULATE(
        DISTINCTCOUNT(Closed_order[Order Reference]),
        USERELATIONSHIP(Closed_order[Closed_Date_Only], 'DimDate'[Date]),
        'DimDate'[Date] = CurrentDate
    )
)
```

---

### 3.5 `Move to Billing`

Counts open orders that moved to billing on the selected date.

```dax
Move to Billing =
VAR CurrentDate = SELECTEDVALUE('DimDate'[Date])
RETURN
IF(
    ISBLANK(CurrentDate),
    BLANK(),
    CALCULATE(
        COUNTROWS(Open_Order),
        USERELATIONSHIP(Open_Order[Effective Billing Date], 'DimDate'[Date]),
        FILTER(
            Open_Order,
            DATEVALUE(Open_Order[Effective Billing Date]) = CurrentDate
                && TRIM(Open_Order[Status]) IN {"Acceptance", "Billing", "On-Hold"}
                && Open_Order[Del Order Coordinator]
                    IN {
                        "Amit VETHEKAR",
                        "Rohan Lokhande",
                        "vaibhav rane",
                        "Akshay CHAPKE",
                        "Ajit Ponkshe"
                    }
        )
    )
)
```

---

### 3.6 `Total Closed (Closed+Billing)`

Equivalent to the original column — currently mirrors `Actual Closed` (extend if billing logic differs).

```dax
Total Closed (Closed+Billing) =
[Actual Closed]
```

---

### 3.7 `Running Open`

Cumulative sum of `Total New Created (Open+Closed)` from 1 Jan 2024 up to the selected date. Returns 0 for future dates.

```dax
Running Open =
VAR CurrentDate = SELECTEDVALUE('DimDate'[Date])
RETURN
IF(
    ISBLANK(CurrentDate) || CurrentDate >= TODAY(),
    0,
    CALCULATE(
        [Total New Created (Open+Closed)],
        FILTER(
            ALL('DimDate'),
            'DimDate'[Date] <= CurrentDate
                && 'DimDate'[Date] >= DATE(2024, 1, 1)
        )
    )
)
```

---

### 3.8 `Running Total Closed (Closed+Billing)`

Cumulative closed count up to the selected date. Returns 0 for future dates.

```dax
Running Total Closed (Closed+Billing) =
VAR CurrentDate = SELECTEDVALUE('DimDate'[Date])
RETURN
IF(
    ISBLANK(CurrentDate) || CurrentDate >= TODAY(),
    0,
    CALCULATE(
        [Total Closed (Closed+Billing)],
        FILTER(
            ALL('DimDate'),
            'DimDate'[Date] <= CurrentDate
                && 'DimDate'[Date] >= DATE(2024, 1, 1)
        )
    )
)
```

---

### 3.9 `Actual Open`

Net open orders: running created minus running closed.

```dax
Actual Open =
[Running Open] - [Running Total Closed (Closed+Billing)]
```

---

### 3.10 `OneOff_Plus_Monthly_EUR_Closed_Factored`

Sum of `One Off Total Gold` from `Closed_order` for the selected closed date.  
*Format: `"₹"\ #,0.00;#,0.00\ -"₹";"₹"\ #,0.00` (bn-IN currency)*

```dax
OneOff_Plus_Monthly_EUR_Closed_Factored =
VAR CurrentDate = SELECTEDVALUE('DimDate'[Date])
RETURN
IF(
    ISBLANK(CurrentDate),
    BLANK(),
    CALCULATE(
        SUM(Closed_order[One Off Total Gold]),
        USERELATIONSHIP(Closed_order[Closed_Date_Only], 'DimDate'[Date]),
        'DimDate'[Date] = CurrentDate
    )
)
```

---

### 3.11 `OneOff_Chgs_EUR_Open_Factored`

Sum of one-off and monthly charges from `Open_Order` for the selected created date, multiplied by the conversion factor 100.351.  
*Format: `"₹"\ #,0;#,0\ -"₹";"₹"\ #,0` (bn-IN currency)*

```dax
OneOff_Chgs_EUR_Open_Factored =
VAR CurrentDate = SELECTEDVALUE('DimDate'[Date])
VAR BaseAmt =
    IF(
        ISBLANK(CurrentDate),
        BLANK(),
        CALCULATE(
            SUM(Open_Order[Tot One Off Chgs Eur])
                + SUM(Open_Order[Tot Month Chgs Eur]),
            Open_Order[Created Date] = CurrentDate
        )
    )
RETURN
    BaseAmt * 100.351
```

---

### 3.12 `Total New Created (All)`

Grand total of daily new created counts across all dates in context (equivalent to `SUM(History_Open[Total New Created(Open+Closed)])`).

```dax
Total New Created (All) =
SUMX(
    VALUES('DimDate'[Date]),
    [Total New Created (Open+Closed)]
)
```

---

### 3.13 `Current Month Total New Created (MTD)`

Month-to-date total of `Total New Created (All)`.

```dax
Current Month Total New Created (MTD) =
TOTALMTD(
    [Total New Created (All)],
    'DimDate'[Date]
)
```

---

### 3.14 `Total Closed Amt`

Sum of `OneOff_Plus_Monthly_EUR_Closed_Factored` across all dates in context.

```dax
Total Closed Amt =
SUMX(
    VALUES('DimDate'[Date]),
    [OneOff_Plus_Monthly_EUR_Closed_Factored]
)
```

---

### 3.15 `Feb_26 Closed Amt`

Closed amount for February 2026 only.

```dax
Feb_26 Closed Amt =
CALCULATE(
    [Total Closed Amt],
    'DimDate'[Month MM-YY] = "Feb-26"
)
```

---

### 3.16 `Mar_26 Closed Amt`

Closed amount for March 2026 only.

```dax
Mar_26 Closed Amt =
CALCULATE(
    [Total Closed Amt],
    'DimDate'[Month MM-YY] = "Mar-26"
)
```

---

### 3.17 `MTD_Closed_Amount`

Month-to-date sum of `OneOff_Plus_Monthly_EUR_Closed_Factored`.

```dax
MTD_Closed_Amount =
TOTALMTD(
    [Total Closed Amt],
    'DimDate'[Date]
)
```

---

### 3.18 `MTD_Open_Amount`

Month-to-date sum of `OneOff_Chgs_EUR_Open_Factored`.

```dax
MTD_Open_Amount =
TOTALMTD(
    SUMX(VALUES('DimDate'[Date]), [OneOff_Chgs_EUR_Open_Factored]),
    'DimDate'[Date]
)
```

---

## Step 4 — Using Measures in Visuals

| Visual Type | Axis / Legend | Values |
|---|---|---|
| Line / Bar chart (daily) | `DimDate[Date]` or `DimDate[Date Text]` | Any measure above |
| Line / Bar chart (monthly) | `DimDate[Month MM-YY]` (sort by `Month Index`) | Any measure above |
| Card | *(no axis)* | `Actual Open`, `Running Open`, `MTD_Closed_Amount`, etc. |
| Table / Matrix | `DimDate[Date]` as row | All daily measures as values |

> **Tip**: Always use `DimDate` columns on the visual axis — **not** `Open_Order` or `Closed_order` date columns directly. This ensures the `SELECTEDVALUE('DimDate'[Date])` pattern inside each measure resolves to a single date.

### Slicer setup

- Add a **Date Range** slicer using `DimDate[Date]`.
- Add a **Month** slicer using `DimDate[Month MM-YY]` (sort by `DimDate[Month Index]`).

---

## Step 5 — Validation: Sample Date Checks

The table below shows expected measure outputs for three sample dates to confirm measures match the old calculated-column outputs.

| Sample Date | `Created (Open File)` | `Actual Closed` | `Running Open` | `Actual Open` |
|---|---|---|---|---|
| 2024-01-15 | Same as old column for that date | Same as old column | Cumulative from 2024-01-01 | Running Open − Running Closed |
| 2024-06-30 | Same as old column for that date | Same as old column | Cumulative from 2024-01-01 | Running Open − Running Closed |
| `TODAY()` | Returns BLANK (future guard) | Returns BLANK (future guard) | 0 (future guard) | 0 (future guard) |

### How to validate in Power BI Desktop

1. Create a **Table visual** with `DimDate[Date]` as rows.
2. Add both the old calculated-column (if still present in model) and the new measure as columns.
3. Scan for rows where values differ — they should all match for past dates.
4. Confirm `Running Open` = 0 for `TODAY()` and future dates.
5. Confirm `MTD_Closed_Amount` resets at the start of each month.

---

## Step 6 — Deleting the Calculated Table

Once you have verified the measures produce correct outputs:

1. Open **Tabular Editor** (or Power BI Desktop Model view).
2. Delete the `History_Open` calculated table.
3. Re-bind any visuals or other measures that referenced `History_Open` columns to the new measures / `DimDate` columns.
4. Save and publish.

---

## Currency Format Strings Reference

| Measure | Format String |
|---|---|
| `OneOff_Plus_Monthly_EUR_Closed_Factored` | `"₹"\ #,0.00;#,0.00\ -"₹";"₹"\ #,0.00` |
| `OneOff_Chgs_EUR_Open_Factored` | `"₹"\ #,0;#,0\ -"₹";"₹"\ #,0` |
| All count measures | `0` |
| `MTD_Open_Amount` | `0` (decimal annotation) |

---

## Summary

All `History_Open` calculated columns and measures are now replaced by the DAX measures defined above. The `DimDate` table acts as the single date scaffold. No calculated table rows are stored in the model, which reduces model size and refresh time while producing identical report outputs.
