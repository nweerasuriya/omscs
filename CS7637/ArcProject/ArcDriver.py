import json
import os.path

import numpy as np
from matplotlib import pyplot as plt
from ArcColors import arc_colors
from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import ArcAgent


def run_training_data(
    agent: ArcAgent, arc_problems: list[ArcProblem]
) -> dict[ArcProblem, tuple[bool, list]]:
    """
    Run each training problem with the test output included so the agent can
    test if they are getting the correct response.
    """
    train_ans_dict: dict[ArcProblem, tuple[bool, list]] = dict()
    for trn_problem in arc_problems:
        preds: list[np.ndarray] = agent.make_predictions(trn_problem)
        correct = False

        if len(preds) <= 3:
            for prediction in preds:
                answer = trn_problem.test_set().get_output_data().data()
                correct = np.array_equal(answer, prediction)
                if correct:
                    break

        # # store the problem_set and whether it was correctly solved
        train_ans_dict[trn_problem] = (correct, preds)

    return train_ans_dict


def load_arc_problems(path: str, problem_data: list[str]) -> list[ArcProblem]:
    problems: list[ArcProblem] = list()
    for problem_name in problem_data:
        with open(os.path.join(path, problem_name)) as p:
            flat_data: dict[str, dict] = json.load(p)
            # convert the data into ArcData (i.e. numpy.ndarray data)
            trn_data: list[ArcSet] = list()
            for dt in flat_data["train"]:
                d_input = ArcData(np.array(dt["input"]))
                d_output = ArcData(np.array(dt["output"]))
                trn_set: ArcSet = ArcSet(arc_input=d_input, arc_output=d_output)
                trn_data.append(trn_set)

            tst_data: list[ArcSet] = list()
            for tst in flat_data["test"]:
                t_input = ArcData(np.array(tst["input"]))
                t_output = ArcData(np.array(tst["output"]))
                tst_set: ArcSet = ArcSet(arc_input=t_input, arc_output=t_output)
                tst_data.append(tst_set)

            arc_problem = ArcProblem(problem_name[:-5], trn_data, tst_data[0])

            # # there should only be one test in the test data
            problems.append(arc_problem)

    return problems


def create_image_from_array(
    test_output: np.ndarray, prediction: np.ndarray, save_path: str
) -> None:
    """
    Create image of input and output for each problem in the milestone data set as pngs
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(test_output, cmap=arc_colors)
    axs[0].set_title("Test Output")
    axs[0].axis("off")

    axs[1].imshow(prediction, cmap=arc_colors)
    axs[1].set_title("Agent Prediction")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":

    # Here you can use this to open other milestone data directories for running against
    #  you'll should copy this code and change the path to the milestone you want to load (B, C or D)
    milestone = "C"
    milestone_path = os.path.join("Milestones", milestone)
    milestone_data: list[str] = os.listdir(milestone_path)

    arc_milestone_problems: list[ArcProblem] = load_arc_problems(
        milestone_path, milestone_data
    )
    # Check if only a single test needs to be run
    test_num = input("If you want to run a single test, enter problem number: ")
    if test_num:
        test_num = int(test_num)
        arc_milestone_problems = [arc_milestone_problems[test_num]]

    # instantiate the agent once
    arc_agent: ArcAgent = ArcAgent()

    milestone_data_set = run_training_data(arc_agent, arc_milestone_problems)
    milestone_file = open(f"Milestone_Results_{milestone}/Milestone_Results.csv", "w")

    print("location of results file: " + os.path.abspath(milestone_file.name))
    milestone_file.write(
        "Problem Name, Correct, Correct Answer, Prediction 1, Prediction 2, Prediction 3\n"
    )
    for i, m_answer_set in enumerate(milestone_data_set.keys()):
        # print("Problem: " + m_answer_set.problem_name())
        m_correct, predictions = milestone_data_set[m_answer_set]
        m_cor_ans = m_answer_set.test_set().get_output_data().data().tolist()
        milestone_file.write(
            f"{m_answer_set.problem_name()}," f"{m_correct}," f'"{m_cor_ans}",'
        )
        if m_correct:
            print("Correctly solved problem: " + m_answer_set.problem_name())
        if len(predictions) == 0:
            milestone_file.write("empty\n")
            continue
        for idx, pred in enumerate(predictions, 1):
            if len(predictions) == idx:
                milestone_file.write(f'"{pred.tolist()}"\n')
            else:
                milestone_file.write(f'"{pred.tolist()}",')
            # Save image
            if idx == 1:  # Only save image for the top prediction
                create_image_from_array(
                    m_answer_set.test_set().get_output_data().data(),
                    pred,
                    os.path.join(
                        f"Milestone_Results_{milestone}",
                        f"images/{m_answer_set.problem_name()}.png",
                    ),
                )

    milestone_file.close()
