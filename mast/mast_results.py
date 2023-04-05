from bs4 import BeautifulSoup
from mast.mast_writer import reverse_output_event_name


def parse_results(file):
    wcrts = {}
    with open(file, "r") as f:
        soup = BeautifulSoup(f, features="xml")

        for task_results in soup.findAll("mast_res:Timing_Result"):
            event_name = (task_results["Event_Name"])
            task_name = reverse_output_event_name(event_name)
            wcrt = find_task_wcrt(task_results)
            wcrts[task_name] = wcrt
    return wcrts


def find_task_wcrt(element):
    wcrt = max([float(x["Time_Value"]) for x in element.findAll("mast_res:Global_Response_Time")])
    return wcrt


if __name__ == '__main__':
    print(parse_results("test-out.xml"))
