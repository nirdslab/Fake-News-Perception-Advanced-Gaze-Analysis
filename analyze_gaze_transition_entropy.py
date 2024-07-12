import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pingouin import mwu
import seaborn as sns


def analyze_entropy_for_version(et_df):
    et_df.drop(et_df[(et_df['believability'] == '-1')].index, inplace=True)

    et_df_true = et_df[et_df['version'] == 'true']
    et_df_fake = et_df[et_df['version'] == 'fake']

    barWidth = 0.7

    bars1 = [et_df_true['Entropy'].mean(), et_df_fake['Entropy'].mean()]
    yer1 = [et_df_true['Entropy'].sem(), et_df_fake['Entropy'].sem()]

    print("True News count: " + str(len(et_df_true)))
    print("Fake News count: " + str(len(et_df_fake)))

    print("Gaze Transition Entropy (mean) ===> ")
    print("True News: " + str(et_df_true['Entropy'].mean()))
    print("Fake News: " + str(et_df_fake['Entropy'].mean()))

    print("Gaze Transition Entropy - SE ===> ")
    print("True News: " + str(et_df_true['Entropy'].sem()))
    print("Fake News: " + str(et_df_fake['Entropy'].sem()))

    print("Gaze Transition Entropy - Sum ===> ")
    print("True News: " + str(et_df_true['Entropy'].sum()))
    print("Fake News: " + str(et_df_fake['Entropy'].sum()))

    fvalue, pvalue = st.f_oneway(et_df_true['Entropy'], et_df_fake['Entropy'])
    print(fvalue, pvalue)

    print("ANOVA Table")
    # Ordinary Least Squares (OLS) model
    model = ols('Entropy ~ C(version)', data=et_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)

    # The x position of bars
    r1 = np.arange(len(bars1))

    f, ax = plt.subplots(figsize=(10, 6))

    # Draw a nested barplot
    g = sns.catplot(
        data=et_df, kind="bar",
        y="Entropy", x="version", palette="Set2", order=['true', 'fake']
    )

    ax.xaxis.grid(True)
    plt.setp(g.ax.lines, linewidth=1)

    g.despine(left=True)
    g.set_axis_labels("Version", "Gaze Transition Entropy")

    plt.savefig(f"Graphs/Entropy/analysis_of_entropy_version.png")

    print("\nAnalyses for Entropy:")
    entropy_true = et_df_true[['Entropy']].to_numpy()
    entropy_fake = et_df_fake[['Entropy']].to_numpy()
    print("Entropy - Normality of true: ", st.shapiro(entropy_true))
    print("Entropy - Normality of fake: ", st.shapiro(entropy_fake))
    print("Entropy - Homogenity of variance: ", st.levene(entropy_true.flatten(), entropy_fake.flatten()))
    print(st.ttest_ind(a=entropy_true, b=entropy_fake, equal_var=True))
    print(mwu(entropy_true, entropy_fake))
    print("Coefficient K - Kruskal", st.kruskal(entropy_true.flatten(), entropy_fake.flatten()))


def analyze_entropy_for_believability(et_df):
    # Box Plot
    import seaborn as sns
    sns.boxplot(et_df['Entropy'])
    plt.show()

    et_df.drop(et_df[(et_df['believability'] == '-1')].index, inplace=True)
    et_df.drop(et_df[(et_df['Entropy'] > 0.8)].index, inplace=True)
    et_df.drop(et_df[(et_df['Entropy'] < - 0.13)].index, inplace=True)

    et_df_believable = et_df[et_df['believability'] == 'believable']
    et_df_unsure = et_df[et_df['believability'] == 'unsure']
    et_df_notbelievable = et_df[et_df['believability'] == 'not believable']

    barWidth = 0.7

    bars1 = [et_df_believable['Entropy'].mean(), et_df_unsure['Entropy'].mean(), et_df_notbelievable['Entropy'].mean()]
    yer1 = [et_df_believable['Entropy'].sem(), et_df_unsure['Entropy'].sem(), et_df_notbelievable['Entropy'].sem()]

    print("Believable count: " + str(len(et_df_believable)))
    print("Unsure count: " + str(len(et_df_unsure)))
    print("Not believable count: " + str(len(et_df_notbelievable)))
    #
    print("Gaze Transition Entropy (mean) ===> ")
    print("Believable: " + str(et_df_believable['Entropy'].mean()))
    print("Unsure: " + str(et_df_unsure['Entropy'].mean()))
    print("Not believable: " + str(et_df_notbelievable['Entropy'].mean()))

    print("Gaze Transition Entropy (SE) ===> ")
    print("Believable: " + str(et_df_believable['Entropy'].sem()))
    print("Unsure: " + str(et_df_unsure['Entropy'].sem()))
    print("Not believable: " + str(et_df_notbelievable['Entropy'].sem()))

    # The x position of bars
    r1 = np.arange(len(bars1))

    plt.bar(r1, bars1, width=barWidth, color='blue', ecolor='black', yerr=yer1, capsize=4)

    f, ax = plt.subplots(figsize=(10, 6))

    # Draw a nested barplot
    g = sns.catplot(
        data=et_df, kind="bar",
        y="Entropy", x="believability", palette="Set2",
        # meanprops={"marker": "|",
        #            "markeredgecolor": "red",
        #            "markersize": "115"}
    )

    ax.xaxis.grid(True)
    plt.setp(g.ax.lines, linewidth=1)

    g.despine(left=True)
    g.set_axis_labels("Believability", "Gaze Transition Entropy")

    # general layout
    plt.xticks([r for r in range(3)], ['believable', 'unsure', 'not believable'])
    plt.xlabel("Believability")
    plt.ylabel('Gaze Transition Entropy ')

    # Show graphic
    # plt.show()
    plt.savefig(f"Graphs/Entropy/analysis_of_entropy_believability.png")

    fvalue, pvalue = st.f_oneway(et_df_believable['Entropy'], et_df_unsure['Entropy'])
    print(fvalue, pvalue)

    fvalue, pvalue = st.f_oneway(et_df_believable['Entropy'], et_df_notbelievable['Entropy'])
    print(fvalue, pvalue)

    fvalue, pvalue = st.f_oneway(et_df_notbelievable['Entropy'], et_df_unsure['Entropy'])
    print(fvalue, pvalue)

    print("ANOVA Table")

    # # Ordinary Least Squares (OLS) model
    model = ols('Entropy ~ C(believability)', data=et_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)

    print("\nAnalyses for Entropy:")
    entropy_believable = et_df_believable[['Entropy']].to_numpy()
    entropy_unsure = et_df_unsure[['Entropy']].to_numpy()
    entropy_notbelievable = et_df_notbelievable[['Entropy']].to_numpy()
    print("Entropy - Normality of believable: ", st.shapiro(entropy_believable))
    print("Entropy - Normality of unsure: ", st.shapiro(entropy_unsure))
    print("Entropy - Normality of not believable: ", st.shapiro(entropy_notbelievable))
    print("Entropy - Homogenity of variance: ", st.levene(entropy_believable.flatten(),
                                                          entropy_unsure.flatten(), entropy_notbelievable.flatten()))
    print("Entropy - Kruskal", st.kruskal(entropy_believable.flatten(), entropy_unsure.flatten(), entropy_notbelievable.flatten()))
    print(mwu(entropy_believable, entropy_unsure))
    print(mwu(entropy_believable, entropy_notbelievable))
    print(mwu(entropy_unsure, entropy_notbelievable))


eyetracking_data_fn = "Data/Advanced_Gaze_Measures/generated_dataset_with_version_and_believability_labels.csv"
et_df = pd.read_csv(eyetracking_data_fn)

analyze_entropy_for_version(et_df)
analyze_entropy_for_believability(et_df)
