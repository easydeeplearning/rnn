import copy, numpy as np
np.random.seed(0)

# calcula a sigmoid para n linearidade
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# calcula a derivada da sigmoid
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# funcionalidades para tratativas com binarios
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

#converte numero binario em decimal
def toDecimal(v):
    out = 0
    for index,x in enumerate(reversed(v)):
        out += x*pow(2,index)
    return out

# parametros iniciais
# ajuste para modificacao do comportamento
rate = 0.1
input_dim_size = 2
hidden_dim_size = 16
output_dim_size = 1


# Inicialização dos pesos aleatoriamente
weights_0 = 2*np.random.random((input_dim_size,hidden_dim_size)) - 1
weights_1 = 2*np.random.random((hidden_dim_size,output_dim_size)) - 1
weights_h = 2*np.random.random((hidden_dim_size,hidden_dim_size)) - 1

# Criação dos vetores pesos*_update para armazenamento da memória de modificacao
weights_0_update = np.zeros_like(weights_0)
weights_1_update = np.zeros_like(weights_1)
weights_h_update = np.zeros_like(weights_h)

    
#apos treinada, calcula a soma de binarios
def calculate(a, b, d):
    for timestep in range(d.size):
        # generate input and output
        X = np.array([[a[d.size - timestep - 1],b[d.size - timestep - 1]]])
        y = np.array([[c[d.size - timestep - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,weights_0) + np.dot(layer_1_values[-1],weights_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,weights_1))

        # decode estimate so we can print it out
        d[d.size - timestep - 1] = np.round(layer_2[0][0])
        

# logica de treinamento. 20000 epocas
for j in range(20000):
#for j in range(10):
    
    # gera um exemplo aleatorio (a + b = c)
    a_int = np.random.randint(largest_number/2) # valor tipo decimal int
    a = int2binary[a_int] # codificado em binario

    b_int = np.random.randint(largest_number/2) # valor tipo decimal int
    b = int2binary[b_int] # codificado em binario

    # resposta verdadeira
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # cria o vetor para armazenamento da resposta
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim_size))
    
    # movimenta ao longo da entrada/saida RNN
    for timestep in range(d.size):
        
        # gera os vetores de entrada/saida
        X = np.array([[a[d.size - timestep - 1],b[d.size - timestep - 1]]])
        y = np.array([[c[d.size - timestep - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,weights_0) + np.dot(layer_1_values[-1],weights_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,weights_1))

        # calculo do erro
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # finalmente, estima a saida
        d[d.size - timestep - 1] = np.round(layer_2[0][0])
        
        # armazena o valor para uso no proximo timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim_size)
    
    for timestep in range(binary_dim):
        
        X = np.array([[a[timestep],b[timestep]]])
        layer_1 = layer_1_values[-timestep-1]
        prev_layer_1 = layer_1_values[-timestep-2]
        
        # erro da output layer
        layer_2_delta = layer_2_deltas[-timestep-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(weights_h.T) + layer_2_delta.dot(weights_1.T)) * sigmoid_output_to_derivative(layer_1)

        # atualizacao dos pesos
        weights_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        weights_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        weights_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    weights_0 += weights_0_update * rate
    weights_1 += weights_1_update * rate
    weights_h += weights_h_update * rate    

    weights_0_update *= 0
    weights_1_update *= 0
    weights_h_update *= 0
    
    # acompanhamento do progresso
    if(j % 1000 == 0):
        print ("Error:" + str(overallError))
        print ("True:" + str(c))
        print ("Prediction:" + str(d))
        out = toDecimal(d)
        print (str(a_int) + " + " + str(b_int) + " = " + str(out))
        print ("------------")

print("------TESTING------")

resp = "y"

while resp == "y":
    a1_int = int(input("Type a1: "))
    a2_int = int(input("Type a2: "))


    result_int = a1_int + a2_int

    a1 = int2binary[a1_int]
    a2 = int2binary[a2_int]
    result = int2binary[result_int]
    d1 = np.zeros_like(result)

    calculate(a1, a2, d1)
    print (str(a1_int) + " + " + str(a2_int) + " = " + str(toDecimal(d1)))
    print ("------------")

