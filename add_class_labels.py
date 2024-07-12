import pandas as pd

participant_stimulus_version = None


def read_processed_features():
    global participant_stimulus_version
    features = pd.read_csv('Data/D2-Processed-features.csv')
    participant_stimulus_version = features[['participant', 'question', 'version', 'believability']]


def get_stimulus_version_read_by_participant(row):
    # Given the row including the participant id and stimulus id, this function will return the version (fake or true)
    # of the stimulus that the given participant has read

    participant_id = int(row["subj"].replace("P", ""))
    stimulus_id = int(row["group"].replace("Stimulus", ""))

    version = participant_stimulus_version.loc[(participant_stimulus_version.participant == participant_id) &
                                               (participant_stimulus_version.question == stimulus_id)].version.item()

    return str(version)


def get_believability_score_of_participant(row):
    # Given the row including the participant id and stimulus id, this function will return the version (fake or true)
    # of the stimulus that the given participant has read

    participant_id = int(row["subj"].replace("P", ""))
    stimulus_id = int(row["group"].replace("Stimulus", ""))

    believability = participant_stimulus_version.loc[(participant_stimulus_version.participant == participant_id) &
                                               (participant_stimulus_version.question == stimulus_id)].believability\
        .item()

    if believability == 5 or believability == 4:
        return "believable"
    elif believability == 3:
        return "unsure"
    elif believability == 2 or believability == 1:
        return "not believable"
    else:
        return believability


def remove_ignore_list_values(df):
    ignore_list = [{'participant': 'P03', 'stimulus': ['Stimulus7', 'Stimulus32']},
                   {'participant': 'P05', 'stimulus': ['Stimulus17']},
                   {'participant': 'P10', 'stimulus': ['Stimulus1', 'Stimulus2']},
                   {'participant': 'P10', 'stimulus': ['Stimulus5', 'Stimulus7', 'Stimulus8', 'Stimulus12',
                                                       'Stimulus15', 'Stimulus19', 'Stimulus24', 'Stimulus28',
                                                       'Stimulus46', 'Stimulus52']},
                   {'participant': 'P11', 'stimulus': ['Stimulus43']},
                   {'participant': 'P12', 'stimulus': ['Stimulus7', 'Stimulus14', 'Stimulus16', 'Stimulus20',
                                                       'Stimulus22', 'Stimulus25', 'Stimulus29', 'Stimulus34',
                                                       'Stimulus35', 'Stimulus41', 'Stimulus43', 'Stimulus45',
                                                        'Stimulus50', 'Stimulus53', 'Stimulus55', 'Stimulus59',
                                                       'Stimulus60']},
                   {'participant': 'P18', 'stimulus': ['Stimulus8']},
                   {'participant': 'P21', 'stimulus': ['Stimulus2']},
                   {'participant': 'P24', 'stimulus': ['Stimulus7', 'Stimulus9']},
                   {'participant': 'P26', 'stimulus': ['Stimulus5', 'Stimulus12', 'Stimulus13', 'Stimulus15',
                                                       'Stimulus17', 'Stimulus20', 'Stimulus23', 'Stimulus27',
                                                       'Stimulus28', 'Stimulus30', 'Stimulus39', 'Stimulus47',
                                                        'Stimulus59', 'Stimulus60']}]

    for p in ignore_list:
        participant = p["participant"]
        for stimulus in p["stimulus"]:
            df.drop(df[(df['subj'] == participant) & (df['group'] == stimulus)].index, inplace=True)

    return df


def add_labels_to_pipeline_output():
    df = pd.read_csv("Data/Advanced_Gaze_Measures/generated_dataset.csv")
    df = remove_ignore_list_values(df)
    df['version'] = df.apply(lambda row: get_stimulus_version_read_by_participant(row), axis=1)
    df['believability'] = df.apply(lambda row: get_believability_score_of_participant(row), axis=1)

    df.to_csv("Data/Advanced_Gaze_Measures/generated_dataset_with_version_and_believability_labels.csv", index=False)


if __name__ == "__main__":
    read_processed_features()
    add_labels_to_pipeline_output()
