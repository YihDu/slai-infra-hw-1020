\ File written by HiGHS .lp file handler
min
 obj: +0.207325546666667 choose_0 +0.27565888 choose_1 +0.39565888 choose_2 
st
 con1: +1 choose_0 +1 choose_1 +1 choose_2 = +1
 con2: +8192 choose_0 +12288 choose_1 +18432 choose_2 <= +32768
bounds
 choose_0 <= 1
 choose_1 <= 1
 choose_2 <= 1
bin
 choose_0
 choose_1
 choose_2
gen
semi
end
