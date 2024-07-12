import pandas as pd
from scipy import stats as st
from pingouin import mwu
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_lhipa_version(et_df):
    print("===== LHIPA - Version =====")
    et_df_true = et_df[et_df['version'] == 'true']
    et_df_fake = et_df[et_df['version'] == 'fake']

    # width of the bars
    barWidth = 0.7

    bars1 = [et_df_true['LHIPA'].mean(), et_df_fake['LHIPA'].mean()]
    yer1 = [et_df_true['LHIPA'].sem(), et_df_fake['LHIPA'].sem()]

    print("True: M = " + str(bars1[0]) + ", SE = " + str(yer1[0]))
    print("Fake: M = " + str(bars1[1]) + ", SE = " + str(yer1[1]))

    # The x position of bars
    r1 = np.arange(len(bars1))

    f, ax = plt.subplots(figsize=(10, 6))

    # Draw a nested barplot
    g = sns.catplot(
        data=et_df, kind="bar",
        y="LHIPA", x="version", palette="Set2", order=['true', 'fake']
    )

    ax.xaxis.grid(True)
    plt.setp(g.ax.lines, linewidth=1)

    g.despine(left=True)
    g.set_axis_labels("News Version", "LHIPA")

    # general layout
    plt.xticks([r for r in range(2)], ['real', 'fake'])
    plt.xlabel("Version")
    plt.ylabel('LHIPA')

    # Show graphic
    # plt.show()
    plt.savefig(f"Graphs/Cognitive_Load/analysis_of_lhipa_version.png")

    # Ordinary Least Squares (OLS) model
    print("ANOVA Table")
    model = ols('LHIPA ~ C(version)', data=et_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)

    print("\nAnalyses for LHIPA:")
    LHIPA_true = et_df_true[['LHIPA']].to_numpy()
    LHIPA_fake = et_df_fake[['LHIPA']].to_numpy()
    print("LHIPA - Normality of true: ", st.shapiro(LHIPA_true))
    print("LHIPA - Normality of fake: ", st.shapiro(LHIPA_fake))
    print("LHIPA - Homogenity of variance: ", st.levene(LHIPA_true.flatten(), LHIPA_fake.flatten()))
    print(st.ttest_ind(a=LHIPA_true, b=LHIPA_fake, equal_var=True))
    print(mwu(LHIPA_true, LHIPA_fake))


def analysis_lhipa_believability(et_df):
    print("===== LHIPA - Believability =====")
    print(len(et_df.groupby(['subj', 'group'])))

    et_df.drop(et_df[(et_df['believability'] == -1)].index, inplace=True)
    # et_df.drop(et_df[(et_df['LHIPA'] > 0.04)].index, inplace=True)

    et_df_believable = et_df[(et_df['believability'] == 'believable')]
    et_df_unsure = et_df[et_df['believability'] == 'unsure']
    et_df_unbelievable = et_df[et_df['believability'] == 'unbelievable']

    # Ordinary Least Squares (OLS) model
    model = ols('LHIPA ~ C(believability)', data=et_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)

    print("Analyzes for LHIPA:")
    lhipa_believable = et_df_believable[['LHIPA']].to_numpy()
    lhipa_unsure = et_df_unsure[['LHIPA']].to_numpy()
    lhipa_unbelievable = et_df_unbelievable[['LHIPA']].to_numpy()
    print("LHIPA - Normality of true: ", st.shapiro(lhipa_believable))
    print("LHIPA - Normality of neutral: ", st.shapiro(lhipa_unsure))
    print("LHIPA - Normality of fake: ", st.shapiro(lhipa_unbelievable))
    print("LHIPA - Homogenity of variance: ",
          st.levene(lhipa_believable.flatten(), lhipa_unsure.flatten(), lhipa_unbelievable.flatten()))
    print("Entropy - Kruskal",
          st.kruskal(lhipa_believable.flatten(), lhipa_unsure.flatten(), lhipa_unbelievable.flatten()))


eyetracking_data_fn = "Data/Advanced_Gaze_Measures/generated_dataset_with_version_and_believability_labels.csv"


et_df = pd.read_csv(eyetracking_data_fn)
analyze_lhipa_version(et_df)
analysis_lhipa_believability(et_df)
