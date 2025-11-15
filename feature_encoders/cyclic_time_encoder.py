from math import sin, cos, pi

class CyclicTimeEncoder:
    HOURS_IN_DAY: int = 24
    MINUTES_IN_HOUR: int = 60
    DAYS_IN_WEEK: int = 7
    MONTHS_IN_YEAR: int = 12

    def __init__(self, minute_in_hour: int, hour_in_day: int, day_in_week: int, month_in_year: int):
        """
        Initialize `CyclicTimeEncoder` object, where receive the minutes, hours, days, and months.
    
        Args:
            minute_in_hour: int → Minutes in hour to encode.
            hour_in_day: int → Hour in day to encode.
            day_in_week: int → Days in week to encode.
            month_in_year: int → Months in year to encode.

        Output:
            None

        Time complexity → O(1)
        """
        sin_hour, cos_hour = self._encode_hour(minute_in_hour)
        sin_day, cos_day = self._encode_day(hour_in_day)
        sin_week, cos_week = self._encode_week(day_in_week)
        sin_month, cos_month = self._encode_month(month_in_year)

        self.encoded_features: dict[str, float] = {
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,

            'sin_day': sin_day,
            'cos_day': cos_day,

            'sin_week': sin_week,
            'cos_week': cos_week,

            'sin_month': sin_month,
            'cos_month': cos_month
        }

    def _encode_hour(self, minute_in_hour: int):
        """
        Encode the given minute coomputing the sin and cos.
    
        Args:
            minute_in_hour: int → Receive the current minute of the hour.

        Output:
            float → Compute sin of the radians for `minute_in_hour`. 
            float → Cos of the radians for `minute_in_hour`.

        Time complexity → O(1)
        """
        hour_radians: float = self._calculate_radians(minute_in_hour, self.MINUTES_IN_HOUR)
        return sin(hour_radians), cos(hour_radians)
    
    def _encode_day(self, hour_in_day: int):
        """
        Encode the given day coomputing the sin and cos.
    
        Args:
            hour_in_day: int → Receive the current hour of the day.

        Output:
            float → Compute sin of the radians for `hour_in_day`. 
            float → Cos of the radians for `hour_in_day`.

        Time complexity → O(1)
        """
        day_radians: float = self._calculate_radians(hour_in_day, self.HOURS_IN_DAY)
        return sin(day_radians), cos(day_radians)

    def _encode_week(self, day_in_week: int):
        """
        Calculate the radians whit a given value and period.
    
        Args:
            value: int → Value to calculate radians in the period window.
            period: int → Full period window.

        Output:
            float →
            float →

        Time complexity → O(1)
        """
        week_radians: float = self._calculate_radians(day_in_week, self.DAYS_IN_WEEK)
        return sin(week_radians), cos(week_radians)
    
    def _encode_month(self, month_in_year: int):
        """
        Encode the given month coomputing the sin and cos.
    
        Args:
            month_in_year: int → Receive the current month of the year.

        Output:
            float → Compute sin of the radians for `month_in_year`. 
            float → Cos of the radians for `month_in_year`.

        Time complexity → O(1)
        """
        adjusted_month: float = month_in_year - 1 # adjust the scale to avoid cyclic overlap
        month_radians: float = self._calculate_radians(adjusted_month, self.MONTHS_IN_YEAR)
        return sin(month_radians), cos(month_radians)
    
    def _calculate_radians(self, value: int, period: int):
        """
        Calculate the radians whit a given value and period.
    
        Args:
            value: int → Value to calculate radians in the period window.
            period: int → Full period window.

        Output:
            None

        Time complexity → O(1)

        Maths:
            (2 * pi * (value)) / period
        """
        return (2 * pi * (value)) / period
    
if __name__ == '__main__':
    """
    Initialize the `CyclicTimeEncoder` object and encode the: minutes, hours, days, and months.

    Time complexity → O(1)

    Initialize → python cyclic_time_encoder.py
    """
    cyclic_time_encoder = CyclicTimeEncoder(minute_in_hour=30, hour_in_day=3, day_in_week=1, month_in_year=1)

    print(cyclic_time_encoder.encoded_features)