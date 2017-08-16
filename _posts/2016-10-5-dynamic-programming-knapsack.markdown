---
title:  "Dynamic Programming: Knapsack Optimization"
date:   2016-10-05
tags: [optimization]

header:
  image: "dynamic_programming_knapsack/vietnam_water_boat.jpg"
  caption: "Photo credit: Ginny Lehman"

excerpt: "Dynamic Programming, Knapsack Problem, Discrete Optimization"
---

One of the quintessential programs in discrete optimization is the [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem). The premise is simple. Given a knapsack with fixed weight capacity and a set of items with associated values and weights:

1. What is the maximum total value we can fit in the knapsack
2. Which items do we put in it to get the maximum total value in the knapsack?

In this post, I'll walk through a standard dynamic programming solution to this problem. 

# What is Dynamic Programming?

According to wikipedia, [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) "is a method for solving a complex problem by breaking it down into a collection of simpler subproblems, solving each of those subproblems just once, and storing their solutions."

So, how do we solve the knapsack problem? By breaking it down into simpler subproblems, solving them, and storing their solutions in a table!

In our case, we'll create a table with columns representing items and rows representing possible knapsack capacities. We'll fill in the table iteratively with the maximum total value for different combinations of capacity and items, and then use the table to find the optimal solution.

# Why Do We Need Dynamic Programming?

We need dynamic programming because the knapsack problem has exponential problem complexity. In this case, we have a complexity of O(2<sup>n</sup>). 

1. 1 item  --> 1 possible knapsack to evaluate.
2. 2 items --> 3 possible knapsacks to evaluate.
3. 3 items --> 7 possible knapsacks to evaluate.
4. 4 items --> 15 possible knapsacks to evaluate.
5. ...
6. n items --> 2<sup>n</sup>-1 knapsacks to evaluate.

Because of this, we quickly lose any ability to check every possible knapsack as the number of items grows. Dynamic programming provides a solution with complexity of O(n * capacity), where n is the number of items and capacity is the knapsack capacity. This scales **significantly** better to larger numbers of items, which lets us solve very large optimization problems such as resource allocation.

Okay, this is pretty abstract. Let's dive in and make it more clear.

# Setting up the Problem

First, I'll define a list of tuples containing values and weights for items. Then I'll define a variable capacity which sets the weight limit for the knapsack. I'll need to keep track of the number items from which I can choose, too.


```python
value_weight_list = [(8, 4), (4, 3), (10, 5), (15, 8)]
capacity = 11
num_items = len(value_weight_list)
```

Great. Now I need to define a blank table to fill in. Based on what I said earlier, the table should have `num_items` number of columns and `capacity` number of rows. However, to simplify the programming logic, I'll actually pad the table with an extra row and column. It'll be clear why this helps when I walk through the code.

I'll also define an empty 3-dimensional array, `intermediate_tables_array`, which I'll use to store every different stage of the table as I fill it in element by element. This will make the dynamic programming process more clear.


```python
import numpy as np
table = np.pad(np.zeros((capacity, num_items)), (0,1), 'constant')
intermediate_tables_array = np.empty((capacity*num_items, capacity+1, num_items+1))
```

# Solving the Optimization Problem

Time to actually solve the problem (fill in the table).


```python
counter = 0
for j in xrange(1, num_items + 1):
    # j is the column in the table
    value = value_weight_list[j-1][0]
    weight = value_weight_list[j-1][1]
    
    for i in xrange(1, capacity + 1):
        # i is the row in the table
        if weight > i:
            table[i,j] = table[i,j-1]
        else:
            table[i,j] = max(table[i,j-1], table[i-weight, j-1] + value)
        
        # not part of the solution (using for illustrative purposes)
        intermediate_tables[counter, :, :] = table
        counter += 1

table
```




    array([[  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   4.,   4.,   4.],
           [  0.,   8.,   8.,   8.,   8.],
           [  0.,   8.,   8.,  10.,  10.],
           [  0.,   8.,   8.,  10.,  10.],
           [  0.,   8.,  12.,  12.,  12.],
           [  0.,   8.,  12.,  14.,  15.],
           [  0.,   8.,  12.,  18.,  18.],
           [  0.,   8.,  12.,  18.,  18.],
           [  0.,   8.,  12.,  18.,  19.]])



Okay. How did we actually populate this optimal value table? Let's break down exactly what the code does.

First, we create a loop to go through each column of the table (each column represents an item in our knapsack). In our case, we want to iterate from `1` to the `num_items + 1` times since we padded our array for easier calculation (now column index 1 represents item 1, column index 2 represents item 2, and so on).


```python
for j in xrange(1, num_items + 1):
    value = value_weight_list[j-1][0]
    weight = value_weight_list[j-1][1]
```

In the loop, I use `j` to loop through every column in our table (the number of items we can choose from). I get the `j-1` item from our weight/value list since we're starting the loop from column index 1 (but I want to start with item 0 -- the first item).


```python
for i in xrange(1, capacity + 1):
```

Now I'm looping through every row in our table (starting from 1) for each column. Similar to the first loop, each row represents a different amount of capacity (row 1 represents capacity = 1, row 2 represents capacity = 2, and so on until the final row which represents the maximum capacity -- 11 in this case). This is a result of padding the table with an extra row.

That's all for the set up. How do we actually populate the table?


```python
if weight > i:
    table[i,j] = table[i,j-1]
```

What's going on here? Since each row index represents the capacity for that row, if the `weight` of our current item is bigger than the index we can't fit it in the backpack. If that's the case, our best action is to just take the same item (or items) we had at this level of capacity before even considering this item. We get that value from the value in the same row one column to the left. If we're looking at the first item (column 1), this value is intuitively zero. The extra column allows me to grab the value of 0 from inside the table (column 0).

But what if we could fit the item in the backpack?


```python
else:
    table[i,j] = max(table[i,j-1], table[i-weight,j-1] + value)
```

If we can fit the item, do we want to take it? Possibly. We want to take the new item if we'd have higher value at this current level of capacity by taking it than not taking it. So, how do we assess that? We've already seen that the `table[i,j-1]` represents the best value at this capacity with the previously seen items.

What does `table[i-weight,j-1] + value` really represent? `table[i-weight,j-1]` is the best value, before looking at the current item, at a capacity level just small enough for us to add in this new item.

To understand how this works, imagine we're looking at the second item in our item set (`j=2`), with value 4 and weight 3.

We've already filled in our table for the first item, with 0 value at capacity less than its weight and 8 value once we can hold it. This is what our table looks like.


    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.]])



When `i < 3`, we clearly can't do better than the maximum value we had from this row looking at the first item (`table[i,1]`). Since we couldn't fit the first item at these capacities either, our maximum value is still 0.

When `i = 3`, we can now possibly take the 2nd item. Our maximum value from the first item in this row is `table[3,1]`, which equals 0 since we couldn't fit the item (it had weight 4). If we made room to take this new item, we'd need to use 3 capacity up. 


So our combined value would be the value at `table[3-3,1]` + the value of this new item. `table[0,1]` is 0, the new value is 4, so our best value when looking at the 2nd item is 4. Thus, `table[3,2]` becomes 4. Our table now looks like this:


    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  4.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.]])



When `i = 4`, we can either take `table[4,1]` or `table[4-3,1]` + the value of item 2. In this case, `table[4,1] = 8` and `table[4-3,1] + value` = 4, so we take the first item instead of the second item, giving us this table:



    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  4.,  0.,  0.],
           [ 0.,  8.,  8.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.],
           [ 0.,  8.,  0.,  0.,  0.]])



As we move through the loop for this item (with increasing capacity size), we continue doing this same assessment and updating our table. After looping through two of the items, we have:


    array([[  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   4.,   0.,   0.],
           [  0.,   8.,   8.,   0.,   0.],
           [  0.,   8.,   8.,   0.,   0.],
           [  0.,   8.,   8.,   0.,   0.],
           [  0.,   8.,  12.,   0.,   0.],
           [  0.,   8.,  12.,   0.,   0.],
           [  0.,   8.,  12.,   0.,   0.],
           [  0.,   8.,  12.,   0.,   0.],
           [  0.,   8.,  12.,   0.,   0.]])



At capacity 7 (i=7), we were able to choose between `table[7,1]` (value of 8) and `table[7-3,1]` + `value of item 2` (which is 8 + 4), since we had the capacity to fit both items. Since 12 > 8, we update our table accordingly.

After doing this same assessment for every item, our best value table is in the bottom right corner.


    array([[  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   4.,   4.,   4.],
           [  0.,   8.,   8.,   8.,   8.],
           [  0.,   8.,   8.,  10.,  10.],
           [  0.,   8.,   8.,  10.,  10.],
           [  0.,   8.,  12.,  12.,  12.],
           [  0.,   8.,  12.,  14.,  15.],
           [  0.,   8.,  12.,  18.,  18.],
           [  0.,   8.,  12.,  18.,  18.],
           [  0.,   8.,  12.,  18.,  19.]])



From the table, it's clear the most value we can carry in the knapsack is 19. But which items give us that amount? We can trace back in the table to find out.

# Finding the Optimal Items

For each item column (all but the 0th index column in the table), starting from the last column, check if the value in the row corresponding to the capacity we have remaining to use is different in the current column and the one before it. If they aren't, it means the item wasn't chosen, so mark the item as such. If they are different, mark the item as chosen and reduce the remaining capacity by the weight of that item. Due to this, the next iteration of the loop will be checking values in the row corresponding to the updated amount of remaining capacity. Continue until all item columns have been looped over.


```python
items_taken = np.empty(num_items).astype(str)
remaining_capacity = capacity

for i in xrange(num_items, 0, -1):
    if table[remaining_capacity][i] != table[remaining_capacity][i-1]:
        items_taken[i-1] = 'Chosen'
        weight = value_weight_list[i-1][1]
        remaining_capacity = remaining_capacity - weight
    else:
        items_taken[i-1] = 'Not Chosen'

np.array(value_weight_list)[ np.where(items_taken == 'Chosen')[0], :]
```




    array([[ 4,  3],
           [15,  8]])



Looks like I put the second and fourth items in the knapsack to get the most value!

# Conclusion

It's pretty clear that dynamic programming can help us find answers to problems that we couldn't otherwise solve. While this 4-item knapsack problem was a toy example, optimization problems are everywhere. Whether it's by allowing us to find the optimal truck routes for a fleet of delivery vehicles, to calculate household consumption and saving to maximize utility in an economic model, or to determine the longest common patterns in sets of DNA sequences, dynamic programming (and other optimization techniques) can help us get answers to problems that impact people's lives.
