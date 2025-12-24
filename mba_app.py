import streamlit as st
import pandas as pd
import numpy as np
from apyori import apriori
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration (UI only)
# --------------------------------------------------
st.set_page_config(
    page_title="Market Basket Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Title Section (UNCHANGED TITLE)
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🛒 Market Basket Analysis</h1>
    <h4 style='text-align:center; color:gray;'>
    Customer Purchase Pattern Discovery using Association Rule Mining
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar – Controls
# --------------------------------------------------
st.sidebar.header("📂 Data & Controls")
st.sidebar.markdown(
    """
    **Step 1:** Upload transaction dataset (CSV)  
    **Step 2:** Select number of words  
    **Step 3:** Choose a product to view recommendations
    """
)

file_bytes = st.sidebar.file_uploader("Upload CSV File", type="csv")
words = st.sidebar.selectbox("Number of words in Word Cloud", range(10, 1000, 10))

# --------------------------------------------------
# Main Layout
# --------------------------------------------------
left_col, right_col = st.columns([1.3, 1.7])

# --------------------------------------------------
# Core Logic (UNCHANGED)
# --------------------------------------------------
if file_bytes is not None:
    dataset = pd.read_csv(file_bytes, header=None)
    rows, cols = dataset.shape

    transactions = []
    for i in range(rows):
        transactions.append([str(dataset.values[i, j]) for j in range(cols)])

    rule_list = apriori(
        transactions,
        min_support=0.003,
        min_confidence=0.1,
        min_lift=3,
        min_length=2
    )

    results = list(rule_list)

    bought_item = [tuple(result[2][0][0])[0] for result in results]
    will_buy_item = [tuple(result[2][0][1])[0] for result in results]
    support_values = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lift_values = [result[2][0][3] for result in results]

    new_data = list(zip(
        bought_item,
        will_buy_item,
        support_values,
        confidences,
        lift_values
    ))

    new_df = pd.DataFrame(
        new_data,
        columns=["Bought Item", "Expected To Be Bought", "Support", "Confidence", "Lift"]
    )

    # --------------------------------------------------
    # Left Panel – Selection
    # --------------------------------------------------
    with left_col:
        st.subheader("🔍 Product Selection")
        st.success("Dataset uploaded successfully")

        st.sidebar.header("Select Item")
        Input = st.sidebar.selectbox(
            "Select a product",
            new_df["Bought Item"].unique()
        )

        st.markdown(
            """
            ℹ️ **Insight:**  
            The system identifies products that are frequently purchased
            together based on historical transaction data.
            """
        )

    # --------------------------------------------------
    # Right Panel – Recommendations
    # --------------------------------------------------
    with right_col:
        st.subheader("📊 Recommendation Insights")

        sample = new_df[new_df['Bought Item'] == Input]

        lis1 = []
        for i in sample["Expected To Be Bought"]:
            lis1.append(i)

        output = " ".join(lis1).replace("nan", "")

        st.markdown(
            f"""
            **Customers who purchased _{Input}_ also frequently purchased:**  
            """
        )

        if output.strip() != "":
            for item in set(lis1):
                if str(item) != "nan":
                    st.markdown(f"✅ {item}")
        else:
            st.warning("No strong association found for the selected product.")

        # --------------------------------------------------
        # Business Insight Section
        # --------------------------------------------------
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("💡 Actionable Business Insight")
        st.markdown(
            f"""
            Customers who buy **{Input}** are likely to buy the recommended items listed above.  
            Retailers can improve cross-selling by:
            - Placing these items together in-store or online  
            - Offering bundle discounts  
            - Recommending them during checkout  
            """
        )

        # --------------------------------------------------
        # Optional: Top Rules Table
        # --------------------------------------------------
        st.subheader("📋 Top Association Rules")
        if len(sample) > 0:
            top_rules = sample.sort_values(by="Confidence", ascending=False).head(5)
            st.dataframe(top_rules.reset_index(drop=True))
        else:
            st.info("No top rules available for this product.")

    # --------------------------------------------------
    # Visualization Section
    # --------------------------------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📈 Purchase Pattern Visualization")

    st.markdown(
        """
        The word cloud below highlights frequently purchased items.  
        Larger words indicate higher purchase frequency.
        """
    )

    wordcloud = WordCloud(
        background_color="white",
        max_words=words
    ).generate(output)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)