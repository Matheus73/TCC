from active_learnig import ActiveLearning

if __name__ == "__main__":
    print("Executando o programa principal")
    active_learning = ActiveLearning()
    try:
        active_learning.run()
    except Exception as e:
        print(str(e))
