
baseline = (108*13*4 + 108*4 + 108) + (108*13*4 + 108*4 + 108)*4

print(baseline)

for t in [1, 2, 5, 10, 20, 50]:
    for h in range(1, 70):
        parameters = h*(h + 12)*4 + h*4 + h  + (h*(h + 12) *4 + h*4 + h)*t
        print("Truncation", t, "Hidden size", h, "Parameters", parameters)

