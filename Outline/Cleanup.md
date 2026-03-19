make the added code in my brach beautiful by following these rules:
1. write extremely simple code, it should be "skimmable" and you should still be able to understand it
2. minimize possible states by reducing number of arguments, remove or narrow any state
3. use discriminated unions to reduce number of states the code can be in
4. exhaustively handle any objects with multiple different types, fail on unknown type 
5. don't write defensive code, assume the values are always what types tell you they are
6. use asserts when loading data, and always be highly opinionated about the parameters you pass around. don't let things be optional if not strictly required
7. remove any changes that are not strictly required
8. bias for fewer lines of code
9. no complex or clever code
10. don't break out into too many function, that's hard to read
11. early returns are great
12. use asserts instead of try catches or default values when you do expect something to exist
13. never pass overrides except strictly necessary, keep argument count low
14. don't make arguments optional if they are actually required
