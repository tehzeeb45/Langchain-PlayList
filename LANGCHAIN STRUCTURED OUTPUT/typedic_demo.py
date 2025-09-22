from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

# Create an instance outside the class definition
new_person: Person = {'name': 'Tehzeeb', 'age': 19}
print(new_person)
