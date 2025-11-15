from json import loads, dumps, dump
from os import getcwd, path

class DuckDB:
    def __init__(self, database_file_name: str, file_dir: str = 'datasets'):
        """
        Initializes the database path and ensures the data file exists.

        Args:
            database_file_name: str → File name.
            file_dir: str → Directory name.

        Output:
            None

        Time complexity → O(1)
        """
        relative_path: str = getcwd()

        self.database_file_name: str = database_file_name
        self.database_path: str = path.join(relative_path, file_dir, database_file_name)
        self._check_if_exists()

    def truncate(self):
        """
        Clears all existing content from the current database file.

        Args:
            None

        Output:
            None

        Time complexity → O(1)
        """
        self._check_if_exists()

        with open(self.database_path, 'w'):
            pass

    def insert(self, to_insert: dict[str, any]):
        """
        Appends a new dictionary (JSON line) to the database file.

        Args:
            to_insert: dict[str, any] → Dictionary containing the data record to be inserted as JSON.

        Output:
            None

        Time complexity → O(1)
        """
        self._check_if_exists()
        
        with open(self.database_path, 'a', encoding='utf-8') as database:
            database.write(f'{dumps(to_insert)}\n')

    def line_by_line(self):
        """
        Reads the database file line by line, yielding dictionary objects.

        Args:
            None

        Output:
            dict[str, any] → A dictionary object parsed from the JSON file line.

        Time complexity → O(l)
        """
        self._check_if_exists()

        with open(self.database_path, 'r', encoding='utf-8') as database:
            for line in database:
                yield loads(line)

    def truncate_and_insert_list(self, list_of_dicts: list[dict]):
        """
        Truncate the entire database, and insert a list of dicts.

        Args:
            list_of_dicts: list[dict] → A list of dicts to insert.

        Output:
            None

        Time complexity → O(l)
        """
        self.truncate()

        with open(self.database_path, 'w', encoding='utf-8') as database:
            dump(list_of_dicts, database, indent=4)

    def _check_if_exists(self):
        """
        Confirms the database file exists, creating it if it is missing.

        Args:
            None

        Output:
            None

        Time complexity → O(1)
        """
        with open(self.database_path, 'a', encoding='utf-8') as database:
            database.write('')

if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Initialize → python duck_db.py
    """
    db = DuckDB('database.duck', 'datasets')

    db.truncate()

    transactions = [
        {"id": "1", "amount": 55.00, "time_diff": 150.5, "velocity": 2.5, "is_fraud": 0},
        {"id": "2", "amount": 2500.99, "time_diff": 2.1, "velocity": 15.8, "is_fraud": 1},
        {"id": "3", "amount": 12.50, "time_diff": 300.0,   "velocity": 1.0, "is_fraud": 0},
        {"id": "4", "amount": 32.00, "time_diff": 500.5, "velocity": 10.0, "is_fraud": 0}
    ]

    for line in transactions:
        db.insert(line)

    for line in db.line_by_line():
        fraud = line['is_fraud']

        if fraud == 1:
            print(line)
            continue