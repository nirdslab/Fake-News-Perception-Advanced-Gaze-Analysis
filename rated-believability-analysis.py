import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
plt.style.use('ggplot')


data_features_path = 'Data/D2-Processed-features.csv'


def read_data_file(filename):
    df = pd.read_csv(filename)
    return df


def plot_rated_believability(df):
    plt.figure(num=None, figsize=(6, 4), dpi=160, facecolor='w', edgecolor='k')
    ax = sns.countplot(data=df[df['believability'] > -1], hue="version", x="believability", palette=['#E67E22', "#2980B9"])
    plt.xlabel('rated believability of news')
    plt.savefig("rated_believability_distribution.jpg")
    plt.show()


def task_performance_analysis(df):
    df = df[df['believability'] > -1]
    true_news = len(df[df['version'] == 'true'])
    fake_news = len(df[df['version'] == 'fake'])

    true_highly_believable = len(df[(df['version'] == 'true') & (df['believability'] == 5)])
    true_believable = len(df[(df['version'] == 'true') & (df['believability'] == 4)])
    true_neutral = len(df[(df['version'] == 'true') & (df['believability'] == 3)])
    true_not_believable = len(df[(df['version'] == 'true') & (df['believability'] == 2)])
    true_highly_not_believable = len(df[(df['version'] == 'true') & (df['believability'] == 1)])

    fake_highly_believable = len(df[(df['version'] == 'fake') & (df['believability'] == 5)])
    fake_believable = len(df[(df['version'] == 'fake') & (df['believability'] == 4)])
    fake_neutral = len(df[(df['version'] == 'fake') & (df['believability'] == 3)])
    fake_not_believable = len(df[(df['version'] == 'fake') & (df['believability'] == 2)])
    fake_highly_not_believable = len(df[(df['version'] == 'fake') & (df['believability'] == 1)])

    real_as_real = (true_highly_believable + true_believable)/true_news
    real_as_unsure = true_neutral/true_news
    real_as_fake = (true_highly_not_believable + true_not_believable)/true_news

    print("Real as Believable: " + str(real_as_real))
    print("Real as Unsure: " + str(real_as_unsure))
    print("Real as Fake: " + str(real_as_fake))

    fake_as_real = (fake_highly_believable + fake_believable) / fake_news
    fake_as_unsure = fake_neutral / fake_news
    fake_as_fake = (fake_not_believable + fake_highly_not_believable) / fake_news

    print("Fake as Believable: " + str(fake_as_real))
    print("Fake as Unsure: " + str(fake_as_unsure))
    print("Fake as Fake: " + str(fake_as_fake))

    print("=================")

    print("Correct Responses (True) %: " + str(real_as_real * 100))
    print("Neutral Response (True) %: " + str(real_as_unsure * 100))
    print("Wrong Responses (True) %: " + str((real_as_fake) * 100))

    print("Correct Responses (Fake) %: " + str(fake_as_fake * 100))
    print("Neutral Response (Fake) %: " + str(fake_as_unsure * 100))
    print("Wrong Responses (Fake) %: " + str((fake_as_real) * 100))

    print("=================")

    print("Correct Responses (True) %: " + str(real_as_real * 100))
    print("Wrong Responses (True) %: " + str((1 - real_as_real) * 100))

    print("Correct Responses (Fake) %: " + str(fake_as_fake * 100))
    print("Wrong Responses (Fake) %: " + str((1 - fake_as_fake) * 100))


df = read_data_file(data_features_path)
plot_rated_believability(df)
task_performance_analysis(df)
