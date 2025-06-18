import pandas as pd


class DataGenerator:
    def __init__(self,
                 num_blocks=50):

        self.num_blocks = num_blocks

    def generate(self, file_path=None):
        columns = ["Block_Name", "Block_ID", "Process_Type", "Ship_Type", "Start_Date", "Duration", "Due_Date",
                   "Workload_H01", "Workload_H02", "Weight", "Length", "Breadth", "Height"]

        df_blocks = []

        for j in range(self.num_blocks):
            name = "J-%d" % j
            id = j

            # 데이터 생성 코드 추가
            process_type =
            ship_type =
            start_date =
            duration =
            due_date =
            workload_h01 =
            workload_h02 =
            weight =
            length =
            breadth =
            height =

            row = [name, id, process_type, ship_type, start_date, duration, due_date,
                   workload_h01, workload_h02, weight, length, breadth, height]

            df_blocks.append(row)

        df_blocks = pd.DataFrame(df_blocks, columns=columns)

        if file_path is not None:
            writer = pd.ExcelWriter(file_path)
            df_blocks.to_excel(writer, sheet_name="blocks", index=False)
            writer.close()

        return df_blocks