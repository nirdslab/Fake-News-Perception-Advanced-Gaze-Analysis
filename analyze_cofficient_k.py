import pandas as pd
from scipy import stats as st
from pingouin import mwu
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def mean_k_analysis(et_df):
    print(len(et_df.groupby(['subj', 'group'])))

    # mean K coefficient
    et_df['mean_K'] = et_df.groupby(['subj', 'group'])['K'].transform('mean')
    et_df = et_df.drop(['timestamp', 'K'], axis=1)
    et_df = et_df.drop_duplicates(keep="last")
    et_df = et_df.reset_index(drop=True)

    et_df.drop(et_df[(et_df['believability'] == '-1')].index, inplace=True)

    et_df_true = et_df[et_df['version'] == 'true']
    et_df_fake = et_df[et_df['version'] == 'fake']

    # seaborn plots
    f, ax = plt.subplots(figsize=(10, 6))

    # Draw a nested barplot
    g = sns.catplot(
        data=et_df, kind="bar",
        y="mean_K", x="version", palette="Set2",
        estimator=np.mean, ci=90, capsize=.2,
    )

    ax.xaxis.grid(True)
    plt.setp(g.ax.lines, linewidth=1)

    g.despine(left=True)
    g.set_axis_labels("Version", "Ambient - Focal Coefficient K (Mean)")

    plt.savefig(f"Graphs/Coefficient_K/analysis_of_mean_K.png")

    # Ordinary Least Squares (OLS) model
    print("ANOVA Table")
    model = ols('mean_K ~ C(version)', data=et_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)

    print("Analyzes for Mean K:")
    k_true = et_df_true[['mean_K']].to_numpy()
    k_fake = et_df_fake[['mean_K']].to_numpy()
    print("Coefficient K - Normality of true: ", st.shapiro(k_true))
    print("Coefficient K - Normality of fake: ", st.shapiro(k_fake))
    print("Coefficient K - Homogenity of variance: ", st.levene(k_true.flatten(), k_fake.flatten()))
    print(st.ttest_ind(a=k_true, b=k_fake, equal_var=True))
    print(mwu(k_true, k_fake))
    print("Coefficient K - Kruskal", st.kruskal(k_true.flatten(), k_fake.flatten()))


def dynamic_k_analysis(et_df):
    new_df = pd.DataFrame(columns=["subj", "group", "timestamp", "K", "version", "believability", "time period"])

    subjs = et_df['subj'].unique()
    groups = et_df['group'].unique()
    for subj in subjs:
        for group in groups:
            df = et_df.loc[(et_df.subj == subj) & (et_df.group == group)]
            # print(subj, group)
            if df.empty:
                print("Empty")
                print(subj, group)
            else:
                if len(df['timestamp']) < 4:
                    print("Timestamp < 4")
                    print(subj, group)
                else:
                    df['time period'] = pd.qcut(df['timestamp'], 4, labels=["T1", "T2", "T3", "T4"])
                    new_df = pd.concat([new_df, df])

    # Dynamic K coefficient
    new_df['mean_K'] = new_df.groupby(['subj', 'group', 'time period'])['K'].transform('mean')
    new_df = new_df.drop(['timestamp', 'K'], axis=1)
    new_df = new_df.drop_duplicates(keep="last")
    new_df = new_df.reset_index(drop=True)

    et_df_true = new_df[new_df['version'] == 'true']
    et_df_fake = new_df[new_df['version'] == 'fake']

    # Line Graph for True News
    Time_Serquence = ["T1", "T2", "T3", "T4"]
    Mean_K_true = []
    SE_K_true = []
    Mean_K_fake = []
    SE_K_fake = []
    for i in range(1, 5):
        ts = "T" + str(i)
        Mean_K_true.append(et_df_true[et_df_true["time period"] == ts].mean_K.mean())
        SE_K_true.append(et_df_true[et_df_true["time period"] == ts].mean_K.sem())
        Mean_K_fake.append(et_df_fake[et_df_fake["time period"] == ts].mean_K.mean())
        SE_K_fake.append(et_df_fake[et_df_fake["time period"] == ts].mean_K.sem())

    print("true:")
    print(Mean_K_true)
    print(SE_K_true)

    print("fake:")
    print(Mean_K_fake)
    print(SE_K_fake)

    # new_df = pd.DataFrame(data, columns=["version", "period", "mean_K"])
    sns.set_theme(style="darkgrid")

    # Plot the responses for different events and regions
    sns.lineplot(x='time period', y="mean_K",
                 hue="version", order=["true", "fake"],
                 data=new_df)

    plt.xlabel('Time Sequence')
    plt.ylabel('Ambient - Focal Coefficient K (Mean)')
    plt.legend()
    plt.savefig(f"Graphs/Coefficient_K/analysis_of_dynamic_k.png")
    plt.show()


def mean_k_analysis_believability(et_df):
    print(len(et_df.groupby(['subj', 'group'])))

    # mean K coefficient
    et_df['mean_K'] = et_df.groupby(['subj', 'group'])['K'].transform('mean')
    et_df = et_df.drop(['timestamp', 'K'], axis=1)
    et_df = et_df.drop_duplicates(keep="last")
    et_df = et_df.reset_index(drop=True)

    et_df.drop(et_df[(et_df['believability'] == '-1')].index, inplace=True)

    et_df_believable = et_df[(et_df['believability'] == 'believable')]
    et_df_unsure = et_df[(et_df['believability'] == 'unsure')]
    et_df_unbelievable = et_df[(et_df['believability'] == 'not believable')]

    f, ax = plt.subplots(figsize=(10, 6))

    # Draw a nested barplot
    g = sns.catplot(
        data=et_df, kind="bar",
        y="mean_K", x="believability", palette="Set2", order=["believable", "unsure", "not believable"],
        estimator=np.mean, ci=90, capsize=.2,
    )
    # # general layout
    plt.xticks([r for r in range(3)], ['believable', 'unsure', 'not believable'])
    plt.xlabel("Believability")
    plt.ylabel('Ambient - Focal Coefficient K (Mean)')

    plt.show()

    plt.savefig(f"Graphs/Coefficient_K/analysis_of_mean_k_believability.png")

    # Ordinary Least Squares (OLS) model
    model = ols('mean_K ~ C(believability)', data=et_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)

    print("Analyzes for Mean K:")
    k_believable = et_df_believable[['mean_K']].to_numpy()
    k_unsure = et_df_unsure[['mean_K']].to_numpy()
    k_unbelievable = et_df_unbelievable[['mean_K']].to_numpy()
    print(k_unbelievable)
    print("Coefficient K - Normality of true: ", st.shapiro(k_believable))
    print("Coefficient K - Normality of neutral: ", st.shapiro(k_unsure))
    print("Coefficient K - Normality of fake: ", st.shapiro(k_unbelievable))
    print("Coefficient K - Homogenity of variance: ",
          st.levene(k_believable.flatten(), k_unsure.flatten(), k_unbelievable.flatten()))
    print("Coefficient K - Kruskal", st.kruskal(k_believable.flatten(), k_unsure.flatten(), k_unbelievable.flatten()))
    print(mwu(k_believable, k_unsure))
    print(mwu(k_believable, k_unbelievable))
    print(mwu(k_unsure, k_unbelievable))


def dynamic_k_analysis_believability(et_df):
    new_df = pd.DataFrame(columns=["subj", "group", "timestamp", "K", "version", "believability", "time period"])

    subjs = et_df['subj'].unique()
    groups = et_df['group'].unique()
    for subj in subjs:
        for group in groups:
            df = et_df.loc[(et_df.subj == subj) & (et_df.group == group)]
            # print(subj, group)
            if df.empty:
                print("Empty")
                print(subj, group)
            else:
                if len(df['timestamp']) < 4:
                    print("Timestamp < 4")
                    print(subj, group)
                else:
                    df['time period'] = pd.qcut(df['timestamp'], 4, labels=["T1", "T2", "T3", "T4"])
                    new_df = pd.concat([new_df, df])

    # Dynamic K coefficient
    new_df['mean_K'] = new_df.groupby(['subj', 'group', 'time period'])['K'].transform('mean')
    new_df = new_df.drop(['timestamp', 'K'], axis=1)
    new_df = new_df.drop_duplicates(keep="last")
    new_df = new_df.reset_index(drop=True)

    new_df.drop(new_df[(new_df['believability'] == '-1')].index, inplace=True)

    # new_df = pd.DataFrame(data, columns=["version", "period", "mean_K"])
    sns.set_theme(style="darkgrid")

    # Load an example dataset with long-form data
    # fmri = sns.load_dataset("fmri")

    # Plot the responses for different events and regions
    sns.lineplot(x='time period', y="mean_K",
                 hue="believability", palette="Set2",
                 data=new_df)

    plt.xlabel('Time Sequence')
    plt.ylabel('Ambient - Focal Coefficient K (Mean)')
    plt.legend()
    plt.savefig(f"Graphs/Coefficient_K/analysis_of_dynamic_k_believability.png")


eyetracking_data_fn = "Data/Advanced_Gaze_Measures/generated_dataset_with_version_and_believability_labels.csv"

et_df = pd.read_csv(eyetracking_data_fn)

mean_k_analysis(et_df)
dynamic_k_analysis(et_df)
mean_k_analysis_believability(et_df)
dynamic_k_analysis_believability(et_df)
