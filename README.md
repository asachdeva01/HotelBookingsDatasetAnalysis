# Hotel Cancellation Analysis
**Course: DSE 501 - Statistics for Data Analysts**

**Institution: Arizona State University**

**Semester: Fall 2025**

## Project Overview
This project analyzes 119,390 hotel booking records from City and Resort hotels to identify factors driving cancellations and provide actionable revenue optimization strategies. We combined statistical hypothesis testing with machine learning to validate insights and build predictive models.

## Business Problem
High cancellation rates cause lost revenue and underutilized room capacity. This analysis identifies cancellation drivers and delivers data-backed policy recommendations for hotel operators.

## My Contributions
**1. Market Segment & Distribution Channel Analysis**

I led the correlation analysis examining how market segments and distribution channels impact cancellations and Average Daily Rate (ADR) across hotel types.

**Key Analyses:**
- Performed Chi-square tests to validate the relationship between market segment and cancellation rates
- Analyzed distribution channel effectiveness (Direct, TA/TO, Corporate, GDS) across City vs. Resort hotels
- Created comparative visualizations showing cancellation rate disparities by booking source
- Identified that Online TA bookings exhibit significantly higher cancellation rates (~37%) compared to Direct bookings (~15%)

**Insights Delivered:**
- Online Travel Agents (OTA) drive volume but carry a higher cancellation risk
- Corporate and Direct bookings show greater reliability for revenue forecasting
- Resort hotels show different channel dynamics than City hotels, requiring segment-specific strategies

**2. Feature Engineering & Pre-Processing**

I contributed to the data preparation pipeline, focusing on creating derived features that enhance model interpretability and predictive power.

**Features Engineered:**
- is_family: Binary flag identifying bookings with children or babies — revealed that family bookings have lower cancellation rates
- duration_of_stay: Combined weekend and weeknight stays for total length analysis
- season_of_booking: Mapped arrival months to seasonal categories (Winter, Spring, Summer, Autumn)

**Pre-Processing Contributions:**
- Handled missing values in children, country, and agent columns
- Assisted in removing high-cardinality and irrelevant columns to streamline the model

**Key Findings:**
- Lead Time: Longer lead times strongly correlate with higher cancellation probability
- Deposit Type: "No Deposit" bookings cancel at significantly higher rates than "Non-Refundable."
- Market Segment: Online TA and Groups show the highest cancellation rates
- Hotel Type: City Hotels experience higher cancellation rates than Resort Hotels
- Family Bookings: Bookings with children/babies show lower cancellation rates


**Business Recommendations:**
- Require deposits for high lead-time bookings (>60 days) and OTA reservations
- Develop segment-specific policies for City Hotel guests
- Incentivize direct bookings to reduce OTA dependency and cancellation risk
- Consider overbooking strategies calibrated by segment and season
