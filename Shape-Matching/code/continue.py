
from train_desc import train_desc
from get_id import get_train_type
import datetime
print('Continue', datetime.datetime.now())

args = get_train_type()
current_time = args.name

print(f"Train Desc: + {current_time}")
train_desc(args, current_time)
