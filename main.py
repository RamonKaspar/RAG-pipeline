import os

from numpy import diff
import tiktoken
from src.database_builder import DatabaseBuilder
from src.pipeline import RAGPipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=True)


def main():
    # Define parameters for database building
    subject = "math"
    embedding_model = "text-embedding-3-large"
    chunk_size = 1000
    overlap_size = 50
    min_text_length = 0

    # Initialize the DatabaseBuilder
    db_builder = DatabaseBuilder(
        subject=subject,
        database=None,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        min_text_length=min_text_length,
    )

    # Uncomment the following line to rebuild the database if necessary
    # db_builder.build_database(root_folder_path="data/")

    # Initialize the RAG pipeline
    pipeline = RAGPipeline(db_builder)

    lecture_content = "Have you ever painted your room in a new, cool color? When you want to paint a room, you obviously need to buy paint for it. But how many liters of paint do you need for a room? On the large paint cans you find in stores, it usually says something like: covers twenty square meters. That's the area you can paint with that amount of paint. To know how much paint you need, you need to know the surface area of the wall. The surface area of a geometric shape is the amount of space enclosed by its boundary. But how do you find the surface area of squares, rectangles, and so on? If you want to find the surface area of a square, you have several options. You could, for example, draw it on graph paper and count the squares. Here's an example of a square with a side length of three. As you can see, the square has nine little squares. Feel free to pause for a moment to count them. But now imagine you want to find the surface area of a huge square this way. The bigger the square, the longer it takes to count all the little squares. That's why it's better to calculate the surface area directly. We do this by breaking the square down into rows of squares. As you can see, we have three identical rows, each with three squares, and thus a total of nine squares. So you can easily calculate the surface area of a square by multiplying the side length by the side length. Normally, you don't calculate in little squares but in meters and square meters. The little squares are then small squares with a side length of one meter. Their area is exactly one meter by one meter, or one square meter. Such a square is also called a unit square. When you divide a square into unit squares, you can determine the surface area by counting the number of squares in the rows and columns and multiplying rows by columns—just like in our example.You can also calculate the surface area of a rectangle. You can use unit squares for this as well, and we proceed in the same way as with the calculation of the surface area of the square. You just need to count the number of squares in each column and row. Then you multiply these numbers together. If the rectangle is, for example, two squares wide and three squares high, you calculate: 2 times 3 = 6. As you can see, you also use the formula here: surface area = length times width. Now you already know how to calculate the surface areas of individual squares and rectangles. But what if your area looks more complicated? Then you can often divide the area into smaller rectangles and squares and add their surface areas at the end. For example: Here you have two squares with a side length of two unit squares, that means an area of four square meters each. Additionally, there is a rectangle with side lengths of five and four unit squares, or an area of twenty square meters. The total surface area is: 4 square meters + 4 square meters + 20 square meters = 28 square meters. This way, you can always determine the surface area when you can divide a shape into individual rectangles or squares."
    #    lecture_content = (
    # """How big is a plot of land? How big is a forest? These questions can often be answered by specifying an area. An area consists of a number and a unit of area. For a rectangle, the area (A) is calculated by multiplying the length by the width. In the next example, the rectangle is 20 cm wide and 50 cm long.
    # The area is A = 20 cm - 50 cm = 1000 cm2. The area is therefore 1000 square centimeters. As you can clearly see here, not only the numbers but also the units are multiplied together. We get 20 - 50 = 1000 and cm - cm = cm2.
    # The following table contains an overview of common units of area, their names and the conversion to another unit of area. Many tasks in maths or physics require area measurements to be converted. For example, when adding areas, all areas must be in the same unit of measurement.
    #
    # Unit Designation Conversion
    # 1 mm2 square millimeter 100 mm2 = 1 cm2
    # 1 cm2 Square centimeter 1 cm2 = 100 mm2
    # 1 dm2 square decimeter 1 dm2 = 100 cm2 = 10,000 mm2
    # 1 m2 square metre 1 m2 = 100 dm2 = 10,000 cm2
    # 1 a Ar 1 a = 100 m2
    # 1 ha hectare 1 ha = 100 a
    # 1 km2 Square kilometer 1 km2 = 100 ha = 1,000 m - 1,000 m
    #
    # As you can see from the conversion table for areas, the conversion factor to the next unit is always 100. To get from one unit to the next, you must therefore either multiply by 100 or divide by 100.
    # If you want to convert from a small unit to a large unit, you can either do this step by step or summaries the steps. You can either multiply by 100 twice (100 - 100) or multiply directly by 10,000. In the next section, we will look at the conversion of area units with examples.
    # Units of area need to be converted in class and in real life. In this section, we will look at examples of how to convert from small to large units of area and vice versa from large units of area to smaller units of area.
    # Let's take an area of 5 square metres. We convert the square metres into ares, hectares and square kilometres by dividing by 100 at each step.
    # However, the 5 square metres can also be converted into smaller area units. For example, it is sometimes interesting to convert square metres to square decimetres (m2 to dm2).
    #
    # In this case, you get from unit to unit by multiplying by 100. For a better understanding, you can see the conversion from square metre to square decimetre, square centimetre and square millimetre here.
    # In the next section, we look at the conversion of area units for decimal numbers (decimal points) with an example.
    # Converting large and small area measurements
    # When calculating with area units, commas sometimes have to be moved. This will be demonstrated with another example. For this task, we assume an area of 1.3246 ares.
    # We first convert the area in ares into hectares and square kilometres. To do this, we move the decimal point two places to the left. This corresponds to a division by 100.
    # The next step is to convert the 1.3246 ares into smaller units of area. To do this, we multiply the number by 100, which shifts the decimal point by 2 places to the right (or zeros are added), i.e. the number increases by a factor of 100.
    # This is how we get from Ar to square metres and square decimetres. This can be continued to square centimetres and square millimetres."""
    # )

    grade_level = 5
    difficulty = 5

    very_easy_string = (
        "Very Easy: Basic recall of facts and concepts. Simple, single-step problems."
    )
    easy_string = "Easy: Simple applications of concepts. Slightly more involved recall and application."
    medium_string = "Medium: Multi-step problems requiring application and understanding of concepts."
    hard_string = "Hard: Complex problems involving higher-order thinking, analysis, and synthesis."
    very_hard_string = "Very Hard: Advanced problems that require evaluation, novel applications, and deep understanding of multiple concepts."

    # map the difficulty level to the string
    difficulty_map = {
        1: very_easy_string,
        2: easy_string,
        3: medium_string,
        4: hard_string,
        5: very_hard_string,
    }

    # Define the user query
    user_query = (
        f"""Please generate exactly one exercise for a student that has just watched a lecture with following lecture:
{lecture_content}
Above you are provided with content from textbooks that can help you generate the exercise.
You can also generate exercises that use concepts in reverse order or in a different context.
The student is in {grade_level}th grade, so use concepts and language appropriate for this age group.
The difficulty level of the exercise should be {difficulty_map.get(difficulty)}.
Output the exercise in the following JSON format:
"""
        + """
  {
    "question": "The actual question or exercise to be solved",
    "solution": {
      "approach": "The approach to solve the question",
        "answer": "The answer to the question"
    },
  },
"""
        + """
- Be sure to just generate one exercise.
- Be sure to use the metric system, if you decide to include units.
- Be very creative and make sure the exercise is engaging and challenging for the student.
- Make sure the extremely hard exercises are extremely hard and require multiple steps to solve,while the easy exercises are very easy to solve and require few steps.
- The exercise should be imaginative and encourage children to use various thinking approaches such as critical thinking, creative thinking and problem-solving.
- DO NOT reference any figure or diagram in the exercise.
"""
    )

    # Define parameters for retrieval
    top_k = 2
    threshold = 0.15

    # Run the pipeline to get the response
    response, tokenUsage = pipeline.run(
        user_query,
        top_k=top_k,
        threshold=threshold,
        max_tokens=500,
        temperature=0.7,
        withRag=True,
    )

    # Print the generated response
    print("Generated Response:", response)
    print("Token Usage:", tokenUsage)


if __name__ == "__main__":
    main()
