https://www.jianshu.com/p/d4cc7be68239

---------------------------

## 一、word2vec训练参数

先根据输入的`train_file`文件创建两个数组，`vocab`和`vocab_hash`，`vocab`是词库数组，一维数组，每一个对象都是`vocab_word`类型；

`vocab_hash`是词库的`hash`表，将词按`hash`映射到词库，`vocab_hash[word_hash]` = 词在词库的位置，在从语料文件建立词库时，对读入的词快速定位到其在词库中的位置。

```C++
struct vocab_word { // 词的结构体，存储包括词本身、词频、哈夫曼编码、编码长度、哈夫曼路径                                                                    
    long long cn;// 词频                                                                                  
    int *point;// 哈夫曼树中从根节点到该词的路径，路径的索引要特别注意，在下面的构建哈夫曼树中会说明                                                                                     
    char *word, *code, codelen;// 分别是：词，哈夫曼编码，编码长度                                                                      
};
```

```
//训练参数格式
-train text8 -output vectors.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary
//训练参数意义
size: 对应代码中layer1_size， 表示词向量的维度，默认值是100。
train: 对应代码中train_file， 表示语料库文件路径。
save-vocab: 对应代码中save_vocab_file, 词汇表保存路径。
read-vocab: 对应代码中read_vocab_file， 表示已有的词汇表文件路径，直接读取，不用从语料库学习得来。
debug: 对应代码中debug_mode， 表示是否选择debug模型，值大于1表示开启，默认是2。开启debug会打印一些信息。
binary: 对应代码中全局变量binary，表示文件保存方式，1表示按二进制保存，0表示按文本保存，默认是0.
cbow: 对应代码中cbow， 1表示按cbow模型训练， 0表示按skip模式训练，默认是1。
alpha: 对应代码中alpha，表示学习率。skip模式下默认为0.025， cbow模式下默认是0.05。
output: 对应代码中output_file， 表示词向量保存路径。
window: 对应代码中window，表示训练窗口大小。默认是5
sample: 对应代码中sample，表示下采样阀值。
hs: 对应代码中hs， 表示按huffman softmax模式训练。默认是0， 表示不使用hs。
negative: 对应代码中negative， 表示按负采样模式训练， 默认是5。值为0表示不采用负采样训练；如果使用，值一般为3到10。
threads: 对应代码中num_threads，训练线程数，一般为12。
iter: 对应代码中iter，训练迭代次数，默认是5.
min-count: 对应代码中min_count，表示最小出现频率，低于这个频率的词会被移除词汇表。默认值是5
classes: 对应代码中classes，表示聚类中心数， 默认是0， 表示不启用聚类。
min-count: read
```

## 二、总体流程

#### main函数

![main函数](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/SourceCode/1_1.png)

#### 训练

![训练](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/SourceCode/1_2.png)

### **2.1 TrainModel（）函数**

```c++
void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  //创建多线程，num_threads 为
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;//初始学习速率

  //词汇文件是否存在，存在则直接读取，不存在则学习，得到vocab和vocab_hash
  if (read_vocab_file[0] != 0){
            printf("ReadVocab\n");
            ReadVocab();
  } else{
            printf("LearnVocabFromTrainFile\n");//词汇文件不存在,read_vocab_file[0]=0
            LearnVocabFromTrainFile();
  }
 if (save_vocab_file[0] != 0) SaveVocab();//根据需要，可以将词表中的词和词频输出到文件
 if (output_file[0] == 0) return;//训练后词向量的输出文件，由binary和classes共同控制输出结果,没有的话直接返回
 InitNet(); // 网络结构初始化
 if (negative > 0) InitUnigramTable();//暂时略过
 start = clock();//开始计时
 //其实网络的实现都是在TrainModelThread中，神经网络分成多线程计算，
 //计算完成之后再进行k-mean聚类。TrainModel生成线程，配置线程。
//TrainModelThread的详细解析在第五部分。
 for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
 //让主线程等待子线程执行结束后，主线程再结束。
 //这样，防止主线程很快执行完后，退出，上一行创建的子线程没有机会执行。  
 for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
 fo = fopen(output_file, "wb");//打开输出文件，
  if (classes == 0) {//不用聚类，输出词向量到文件中
    // 首先打印出词汇表的大小，再打印出词向量维度
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);//按照词汇表输出词汇
     //binary: 对应代码中全局变量binary，表示文件保存方式，1表示按二进制保存，0表示按文本保存，默认是0.
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } 
}
```

>
>
>```
>vocab_size: 词汇表大小
>
>layer1_size: 词向量长度
>
>TrainModelThread: 网络实现，训练
>```

## 三、构建词库和hash

word2vec支持两种数据，一种是已经统计好词频的形式，如“中国 6 地球 8”；另外一种是只做了分词，并没有统计词频的形式，如“我 爱 北京 天安门”。在`TrainModel()`这个函数中我们可以看到

```c++
// 如果是带有频率的词表则读入词表和频率，否则读词表并统计频率 
if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
```

### **3.1 LearnVocabFromTrainFile() 这个函数是读取不带频率的文件所用的**

从训练文件中获取所有词汇并构建词表和hash

```c++
/*相关变量说明*/
 const int vocab_hash_size = 30000000;  
 int *vocab_hash;//词库的hash表，将词按hash映射到词库，vocab_hash[word_hash] = 词在词库的位置，
//在从语料文件建立词库时，对读入的词快速定位到其在词库中的位置，训练时不会用到

//从训练文件中获取所有词汇并构建词表和hash
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];//MAX_STRING为一个word的最大长度
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;// 0~30000000
  fin = fopen(train_file, "rb");//打开训练文件 train_file（作为训练参数输入）
  if (fin == NULL) {//找不到该文件，直接退出
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;//词库中实际的词个数,初始为0
  // 将</s>添加到词表最开始的位置
  AddWordToVocab((char *)"</s>");//添加字符</s>，该函数的定义请看4.2部分，添加一个单词到词表尾部，并返回该单词所在的位置
  while (1) {
    ReadWord(word, fin);//ReadWord（）函数参看4.4部分,读取训练文件
    if (feof(fin)) break;//读取到文件末尾
    //读到每一个word
    train_words++;//要训练的词总数
    // debug_mode模式打印进度，暂时不管
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);// 查找词在词库中的位置，词库中存在该词，返回词的位置，否则返回-1,参看4.5部分
    if (i == -1) {// 不在词库中
      a = AddWordToVocab(word);// 添加进去
      vocab[a].cn = 1;// 词频唯一
    } else vocab[i].cn++;// 不用添加，增加词频即可
      /**
         * 如果词表的规模大于哈希空间的70%，则移除一部分低频词
         * 每添加一个词，判断一次词库大小，若大于填充因子上限，删除一次低频词
         * 注意，这里有一个问题，在删除低频词时，语料文件可能是未处理完的，只是读取了一部分词
         * 所以词库当前状态下的词频信息是局部的，不是训练文件全局的
         * 这时删除低频词时，是把局部的低频词删除，但局部低频词未必是全局低频词，例如，一个词在训练文件的前一部分少量出现，但在后面部分，大量出现
         * 不过考虑到词库规模（千万级），这个问题也不太可能出现
         */
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();//去除低频词，参看4.6部分
  }
SortVocab();// 把词表中的单词按词频排序 ，参看4.7部分
 if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);//训练文件大小，ftell得到，多线程训练时会对文件进行分隔，
  //用于定位每个训练线程开始训练的文件位置
  printf("file_size: %lld\n", file_size);
  fclose(fin);
}
```

> ```
> const int vocab_hash_size = 30000000;
> ```
>
> 如果词表的规模大于哈希空间的70%，则移除一部分低频词
> 每添加一个词，判断一次词库大小，若大于填充因子上限，删除一次低频词
>
> 注意，这里有一个问题，在删除低频词时，语料文件可能是未处理完的，只是读取了一部分词
> 所以词库当前状态下的词频信息是局部的，不是训练文件全局的
> 这时删除低频词时，是把局部的低频词删除，但局部低频词未必是全局低频词，例如，一个词在训练文件的前一部分少量出现，但在后面部分，大量出现
> 不过考虑到词库规模（千万级），这个问题也不太可能出现

### **3.2 AddWordToVocab()是添加一个单词到词表尾部**

```c++
//词表结构
struct vocab_word *vocab;
vocab = (struct vocab_word *)calloc(vocab_max_size,  sizeof(struct vocab_word));
struct vocab_word {
  long long cn;//词频，从训练集中计数得到或直接提供词频文件
  int *point;//Haffman树中从根节点到该词的路径，存放的是路径上每个节点的索引
  //word为该词的字面值,code为该词的haffman编码,codelen为该词haffman编码的长 度
  char *word, *code, codelen;
};
```

```c++
/*相关变量说明*/
long long vocab_max_size = 1000;//vocab_max_size词库规模（词库容量），在建立词库的过程中，
//当词库规模到达vocab_max_size时会对词库扩容，每次扩增vocab_max_size个容量
//for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;// 0~30000000

//添加一个单词到词表尾部
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;//保证单词的长度不超过MAX_STRING
  // vocab_size为词库中实际的词个数
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;//词频暂时记为0
  vocab_size++;//词个数增加
 // 如果词表快到上限了，为词表重新申请空间
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);//获取word的hash值，GetWordHash()请看4.3部分
  printf("%lld\n",hash);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;//30000000;开放定址法解决冲突，
 //哈希冲突时线性探测继续顺序往下查找空白位置;vocab_hash[hash] != -1,则说明已经有这个hash了
  vocab_hash[hash] = vocab_size - 1;// 记录在词汇表中的存储位置
  return vocab_size - 1;// 返回添加的单词在词汇表中的存储位置，末尾前一个
}
```

>```c++
>long long vocab_max_size = 1000;//vocab_max_size词库规模（词库容量），在建立词库的过程中，
>//当词库规模到达vocab_max_size时会对词库扩容，每次扩增vocab_max_size个容量
>```

### **3.3 GetWordHash()是添加一个单词到词表尾部**

```c++
//对一个单词进行哈希得到它的哈希值 
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;//打印long long类型：printf("%lld\n",word[1]);
/*计算hash值*/
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;//词库大小取模
  return hash;
}
```

### **3.4 ReadWord()是从文件中读取单个单词**

```c++
// 从文件中读取单个单词，假设单词之间通过空格或者tab键或者EOL键进行分割的
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;//a - 用于向word中插入字符的索引；ch - 从fin中读取的每个字符

  //整个循环就是为了读取一个单词
  while (!feof(fin)) {// 如果fp文件指针没有到达文件尾
    ch = fgetc(fin);// 读取一个词,准确地说是一个字符
    if (ch == 13) continue;
    // 回车，开始新的一行，重新开始while循环读取下一个字符
    // 当遇到space(' ') + tab(\t) + EOL(\n)时，认为word结束，UNIX/Linux中‘\n’为一行的结束符号，
    // windows中为：“<回车><换行>”，即“\r\n”；Mac系统里，每行结尾是“<回车>”,即“\r”
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {//word结束
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);//退回到流中
        break;//如果读到了单词就直接退出
      }
      if (ch == '\n') {//最后还是换行符，说明文件为空，直接退出
        strcpy(word, (char *)"</s>");//将</s>赋予给word
        return;
      } else continue;//此时a＝0，且遇到的为\t or ' '，直接跳过取得下一个字符
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // MAX_STRING为一个word的最大长度，超过了就直接截断
  }
  word[a] = 0;//最后一个字符是'\0'
  printf(word);
  putchar('|');
}
```

### **3.5 SearchVocab()查找词在词库中的位置，词库中存在该词，返回词的位置，否则返回-1**

```c++
// 查找词在词库中的位置，词库中存在该词，返回词的位置，否则返回-1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);//获取hash值
  while (1) {
    if (vocab_hash[hash] == -1) return -1;//词库中不存在该词
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];//vocab_hash[hash] ！= -1且word一样，直接返回位置
    hash = (hash + 1) % vocab_hash_size;//vocab_hash[hash] ！= -1且word不一样，冲突策略
  }
  return -1;
}
```

### **3.6 ReduceVocab() 删除低频词**

```c++
/*变量说明*/

// 从语料文件建立词库时，为了控制词库规模，会在词库的hash表达到填充因子上限时，调用该方法删除一些低频词
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
 //vocab_size为词库里的实际词汇数
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
 //移位
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;//实际词汇数变了
  //处理hash映射
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;//重新全部为-1
  for (a = 0; a < vocab_size; a++) {//实际词汇数
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);//重新计算hash
   //冲突了就重新寻址
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;//没冲突这就是位置
  }
  fflush(stdout);
  min_reduce++;//阈值变大
}
```

### **3.7 SortVocab() 把词表中的单词按词频排序**

```c++
// 比较两个词的词频大小，用于对词库排序
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// 把词表中的单词按词频排序 
void SortVocab() {
  int a, size;
  unsigned int hash;
  // 快排排序，按照词频从大到小排
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;//清楚原来的hash映射
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
     // 如果当前单词词频小于规定的阈值，则删除词表最后一个词 （因为已经从大到小排序了）
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[vocab_size].word); // 删除最后一个词
    } else {
      // 排序以后词表的序改变了，需要重新计算哈希值 
      hash = GetWordHash(vocab[a].word); // 计算hash值
      // 说明hash冲突了，线性寻址
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a; // 找到位置
      train_words += vocab[a].cn; // 重新计算train_words 
    }
  }
  // 因为删了一些词，所以重新定义词表大小 
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  // 给每一个词汇的Huffman编码和路径申请空间 
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}
```

### **3.8 ReadVocab() 读取词频文件，过程和LearnVocabFromTrainFile差不多**

```c++
// 如果是带有频率的词表则读入词表和频率，否则读词表并统计频率 
if (read_vocab_file[0] != 0) ReadVocab(); 
else LearnVocabFromTrainFile();
// 因此ReadVocab的过程和LearnVocabFromTrainFile差不多

// 从词汇表文件中读词并构建词表和hash表
// 由于词汇表中的词语不存在重复，因此与LearnVocabFromTrainFile相比没有做重复词汇的检测
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
 //打开词汇表文件
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
 //初始化hash词表
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;//实际词汇数初始化
//开始处理词汇表文件
  while (1) {
    ReadWord(word, fin);// 从文件中读入一个词
    if (feof(fin)) break;
    //将该词添加到词表中，创建其在hash表中的值，并通过输入的词汇表文件中的值 
    //来更新这个词的词频
    //不存在重复的问题，所以不用search
    a = AddWordToVocab(word);//返回在词汇表中的位置
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);//读取词频，并设置词频
    i++;
  }
  SortVocab();//排序
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");// 还得打开以下训练文件好知道文件大小是多少
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);//将文件指针移至文件末尾
  file_size = ftell(fin);//得到训练文件大小
  fclose(fin);
}
```

### **3.9 SaveVocab() 输出单词和词频到文件（Vocab.txt）**

```c++
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}
```

## 四、初始化网络结构

### **4.1 初始化网络参数**

```c++
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  // syn0存储的是词表中每个词的词向量, 源码中使用一个real(float)类型的一维数组表示，注意是一个一维数组！
  // 容量大小为vocab_size * layer1_size(词向量的维度大小,也是隐藏层的大小)，即 词汇量 * 词向量维度
  // 调用posiz_memalign来为syn0分配内存，对齐的内存，大小为vocab_size * layer1_size * sizeof(real),
  // 也就是每个词汇对应一个layer1_size的向量
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}//没有分配到内存，退出程序
  if (hs) {//基于hierarchical softmax的模型
    //syn1存储的是Huffman树中每个非叶节点的向量
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {
      printf("Memory allocation failed\n"); exit(1);}//未分配到内存，退出程序
    for (a = 0; a < vocab_size; a++) 
      for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;// 初始化syn1为0,初始化非叶节点的向量为0
}
  if (negative > 0) {// 基于negative Sampling模型
   // 负采样时，存储每个样本对应的词向量，一维数组，第i个词的词向量
   // 为syn1neg[i * layer1_size, (i + 1) * layer1_size - 1]
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}//未分配到内存，退出程序
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
 //初始化词向量syn0，每一维的值为[-0.5, 0.5]/layer1_size范围内的随机数
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree(); //创建Haffman二叉树
}
```

>syn0存储的是词表中每个词的词向量, 源码中使用一个`real(float)`类型的一维数组表示，注意是一个一维数组！容量大小为`vocab_size * layer1_size`(词向量的维度大小,也是隐藏层的大小)，即 **词汇量 * 词向量维度调用posiz_memalign来为syn0分配内存，对齐的内存，大小为**`vocab_size * layer1_size * sizeof(real)`**,也就是每个词汇对应一个layer1_size的向量**
>
>- 基于negative Sampling模型 : 负采样时，存储每个样本对应的词向量，一维数组，第i个词的词向量
>- 基于hierarchical softmax的模型 : syn1存储的是Huffman树中每个非叶节点的向量
>
>初始化词向量`syn0`，每一维的值为$\frac{[-0.5,0.5]}{layer1\_size}$范围内的随机数

### **4.2 构建哈夫曼树**

 哈夫曼树

 一般得到霍夫曼树后我们会对叶子节点进行霍夫曼编码，由于权重高的叶子节点越靠近根节点，而权重低的叶子节点会远离根节点，这样我们的高权重节点编码值较短，而低权重值编码值较长。这保证的树的带权路径最短，也符合我们的信息论，即我们希望越常用的词拥有更短的编码。如何编码呢？

**在word2vec中，约定编码方式和上面的例子相反，即约定左子树编码为1，右子树编码为0，同时约定左子树的权重不小于右子树的权重。在有n个叶子节点的哈夫曼树中，其节点总数为2n-1。**

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/SourceCode/1_3.png)

#### 哈夫曼编码

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/SourceCode/1_4.png)

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/SourceCode/1_5.png)	

```c++
 // 利用统计到的词频构建Haffman二叉树
 // 根据Haffman树的特性，出现频率越高的词其二叉树上的路径越短，即二进制编码越短
void CreateBinaryTree() {
 // 用来暂存一个词到根节点的Haffman树路径
 // MAX_CODE_LENGTH是point域和code域大小 ，路径长度其实就是haffman编码 
的函数
 // int *point; 霍夫曼树中从根节点到该词的路径，存放路径上每个非叶结点的索引
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  // 用来暂存一个词的Haffman编码
  // 内存分配，Haffman二叉树中，若有n个叶子节点，则一共会有2^(n-1)个节点 
  // count数组前vocab_size个元素为Haffman树的叶子节点，初始化为词表中所有词的词频, count数组后vocab_size个元素为Haffman书中即将生成的非叶子节点（合并节点）的词频，初始化为一个大值1e^15
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
 //binary数组中前vocab_size存储的是每一个词的对应的二进制编码，后面初始化的是0，用来存储生成节点的编码
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
 //parent_node数组中前vocab_size存储的是每一个词的对应的父节点，后面初始化的是0，用来存储生成节点的父节点
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
 //count数组的初始化
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
 //以下部分为创建Haffman树的算法，默认词表已经按词频由高到低排序
  //</s>词也包含在树内
  pos1 = vocab_size - 1;//末尾
  pos2 = vocab_size;//末尾后面一个
  //构建霍夫曼树，最多进行vocab_size-1次循环操作，每次添加一个节点，即可构成完整的树
  for (a = 0; a < vocab_size - 1; a++) {
    // 'min1, min2'分别用于存储最小和次小节点,注意vocab中的词是已经按照cn排好序的了,是按照降序排列的
    // pos1表示取最原始的词对应的词频,而pos2表示取合并最小值形成的词频
    // 连续两次取,两次取的时候代码操作时一模一样的
    // 第一个if查找最小的权重位置
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) { // 如果count[pos1]比较小，则pos1左移，反之pos2右移
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {//pos1<0,说明叶子节点已经被合并完了，只能往右边找非叶子节点进行合并
      min1i = pos2;
      pos2++;
    }
   //第二个if查找最小的权重位置
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) { //如果count[pos1]比较小，则pos1左移，反之pos2右移
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {//pos1<0,说明叶子节点已经被合并完了，只能往右边找非叶子节点进行合并
      min2i = pos2;
      pos2++;
    }
   //在count数组的后半段存储合并节点的词频（即最小count[min1i]和次小 count[min2i]词频之和）
    count[vocab_size + a] = count[min1i] + count[min2i];//a
    parent_node[min1i] = vocab_size + a;//父节点的位置
    parent_node[min2i] = vocab_size + a;//父节点的位置
    binary[min2i] = 1;//词频较大的为1，左为1，右为0（因为有组合，所以有编码）
  }
 // 建好了hufuman树之后，就需要分配code了，注意这个hufuman树
 //是用一个数组来存储的，并不是我们常用的指针式链表
 // 顺着父子关系找回编码,vocab_size个词汇
  for (a = 0; a < vocab_size; a++) {
    b = a;//从第一个叶子节点开始
    i = 0;
   //找到一个word的huffman编码
    while (1) {
      code[i] = binary[b];// 这个位置对应的编码
      point[i] = b;// 在point数组中增加路径节点的编号
      i++;// Haffman编码的当前长度，从叶子结点到当前节点的深度
      b = parent_node[b];// 找到父节点位置
      if (b == vocab_size * 2 - 2) break; // 由于Haffman树一共有vocab_size*2-1个
      // 节点，所以vocab_size*2-2为根节点
    }
    // 以下要注意的是，同样的位置，point总比code深一层
    vocab[a].codelen = i;//编码长度为i，编码长度赋值，少1，没有算根节点
   // Haffman编码和路径都应该是从根节点到叶子结点的，因此需要对之前得到的code和point进行反向
    vocab[a].point[0] = vocab_size - 2;// // 逆序，把第一个赋值为root（即2*vocab_size - 2 - vocab_size）
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];//逆序，换成vocab[a].code[i - 1-b] = code[b]更 好理解
// 路径逆序，point的长度比code长1，即根节点，point数组“最后”一个值是负的，point路径是为了定位节点位置，叶子节点即是词本身，不用定位，所以训练时这个负数是用不到的
      vocab[a].point[i - b] = point[b] - vocab_size; // 注意这个索引对应的是huffman树中的非叶子节点，对应syn1中的索引， 因为非叶子节点都是在vocab_size * 2 + 1 的后(vocab_size + 1)个
    }
  }
  // 释放内存，point、code、codelen都已经得到
  free(count);
  free(binary);
  free(parent_node);
}
```

霍夫曼树的建立其实并不难，过程如下：

　　　　输入：权值为$(w_1,w_2,\cdots,w_n)$的$n$个节点

　　　　输出：对应的霍夫曼树

　　　　1. 将$(w_1,w_2,\cdots,w_n)$看做是有$n$棵树的森林，每个树仅有一个节点。
   　　　　2. 在森林中选择根节点权值最小的两棵树进行合并，得到一个新的树，这两颗树分布作为新树的左右子树。新树的根节点权重为左右子树的根节点权重之和。
   　　　　3. 将之前的根节点权值最小的两棵树从森林删除，并把新树加入森林。
   　　　　4. 重复步骤2）和3）直到森林里只有一棵树为止。

# 五、模型训练

## 5.1 预生成expTable

word2vec计算过程中用上下文预测中心词或者用中心词预测上下文，都需要进行预测；而word2vec中采用的预测方式是逻辑回归分类，需要用到sigmoid函数，具体函数形式为:
$$
\sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x}
$$
在训练过程中需要用到大量的sigmoid值计算，如果每次都临时去算$e^x$的值，将会影响性能；当对精度的要求不是很严格的时候，我们可以采用近似的运算。在word2vec中，将区间$[-MAX\_EXP, MAX\_EXP]$($[−6,6]$)等距划分为EXP_TABLE_SIZE(默认值1000)等份，并将每个区间的sigmoid值计算好存入到expTable中。在需要使用时，只需要确定所属的区间，属于哪一份，然后直接去数组中查找。expTable初始化代码如下:

```c++
expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));       
//初始化expTable，近似逼近sigmoid(x)值，x区间为[-MAX_EXP, MAX_EXP]，分成EXP_TABLE_SIZE份
//将[-MAX_EXP, MAX_EXP]分成 EXP_TABLE_SIZE 份
for (i = 0; i < EXP_TABLE_SIZE; i++) {
  // 注意：在代码中，作者使用的是小于EXP_TABLE_SIZE，实际的区间是[−6,6)[−6,6)
  expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);   // Precompute the exp() table
  expTable[i] = expTable[i] / (expTable[i] + 1);                     // Precompute f(x) = x / (x + 1)
}
```

### **5.2 CBOW模型**

在CBOW模型中，总共有三层，分别是输入层，映射层和输出层。如下图所示:

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/SourceCode/1_6.png)

hs模式和negative模式中，输入层到映射层的处理是一样的，仅仅是映射层到输出层的处理不一致。

**输入层到映射层的具体操作是：将上下文窗口中的每个词向量求和，然后再平均，得到一个和词向量一样维度的向量，假设叫上下文向量，这个向量就是映射层的向量。**

## 5.3 Skip-Gram模型

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/SourceCode/1_9.png)

## 5.4 TrainModelThread()函数，实现神经网络，进行训练

```c++
// 实现神经网络，进行训练
void *TrainModelThread(void *id) {
  // word: 在提取句子时用来表示当前词在词表中的索引 , 也就是说向sen中添加单词用 , 句子完成后表示句子中的当前单词
 // last_word: 上一个单词，辅助扫描窗口，记录当前扫描到的上下文单词
 // sentence_length: 当前处理的句子长度，当前句子的长度（单词数）
 // sentence_position: 当前处理的单词在当前句子中的位置（index）
 // cw: 窗口长度（中心词除外）
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
 // word_count: 当前线程当前时刻已训练的语料的长度
 // last_word_count: 当前线程上一次记录时已训练的语料长度
 // last_word_count：保存值，以便在新训练语料长度超过某个值时输出信息
 // sen 单词数组，表示句子，//sen：当前从文件中读取的待处理句子，存放的是每个 词在词表中的索引
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
 // l1：在skip-gram模型中，在syn0中定位当前词词向量的起始位置
 // l2：在syn1或syn1neg中定位中间节点向量或负采样向量的起始位置
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g; // f e^x / (1/e^x)，fs中指当前编码为是0（父亲的左子节点为0，右为1）的概率，ns中指label是1的概率
  // g 误差(f与真实值的偏离)与学习速率的乘积
  clock_t now; // 当前时间，和start比较计算算法效率
  //neu1：输入词向量，在CBOW模型中是Context(x)中各个词的向量和，在skip-gram模型中是中心词的词向量
  real * neu1 = (real *)calloc(layer1_size, sizeof(real));
  //neuele：累计误差项
  real * neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE * fi = fopen(train_file, "rb");//打开训练文件
 // 多线程模型训练：利用多线程对训练文件划分，每个线程训练一部分的数据
 // 每个进程对应一段文本，根据当前线程的id找到该线程对应文本的初始位置
 // file_size就是之前LearnVocabFromTrainFile和ReadVocab函数中获取的训练文件的大小
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET); // 将文件内容分配给各个线程
  while (1) { 
  // 对每一个词，应用四种模型进行训练
  // 每训练10000个词时，打印已训练数占所有需要训练数比例，以及打印训练时间；然后更新学习率
    if (word_count - last_word_count > 10000) {
  // word_count表示当前线程当前时刻已 训练的语料的长度
  // last_word_count当前线程上一次记录时已训练的语料长度
  // word_count_actual是全局变量
      word_count_actual += word_count - last_word_count;// word_count_actual是所有线程总共当前处理的词数
      last_word_count = word_count;// 更新last_word_count的值
        // debug_mode 大于0，加载完毕后输出汇总信息，大于1，加载训练词汇的时候输出信息，训练过程中输出信息
      if ((debug_mode > 1)) {// debug模式下输出一些训练信息
        // 输出信息包括：
        // 当前的学习率alpha；
        // 训练总进度（当前训练的总词数/(迭代次数*训练样本总词数)+1）；
        // 每个线程每秒处理的词数
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      //在初始学习率的基础上，随着实际训练词数的上升，逐步降低当前学习率（自适应调整学习率）
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));//自动调整学习率
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;// 调整的过程中保证学习率不低于starting_alpha * 0.0001
    }//以上一整块都在基础配置阶段

    // 从训练样本中取出一个句子，句子间以回车分割（只取出一个句子，因为外面还有一层循环）
    if (sentence_length == 0) {// 如果当前句子长度为0 ，从训练样本中取出一个句子，句子间以回车分割
    // sentence_length被初始值为0，所以一定进入该语句块
      while (1) {
        // word用来表示当前词在词表中的索引
        word = ReadWordIndex(fi);
        // 从文件中读入一个词，并返回这个词在词汇表中的位置，关于该函数，请参看5.2部分。
        if (feof(fi)) break; // 文件结束，退出最近的循环
        if (word == -1) continue;
        word_count++;// 当前线程当前时刻已训练的语料的长度
        if (word == 0) break;// 如果读到的时回车，表示句子结束，退出当前循环，读下一个句子，</s>的索引为0
         // 这里的亚采样是指 Sub-Sampling，Mikolov 在论文指出这种亚采样能够带来 2 到 10 倍的性能提升，并能够提升低频词的表示精度。
        // 低频词被丢弃概率低，高频词被丢弃概率高
        // 对高频词进行随机下采样，丢弃掉一些高频词，能够使低频词向量更加准确，同时加快训练速度
        //可以看作是一种平滑方法
        if (sample > 0) {// sample 亚采样概率的参数，亚采样的目的是以一定概率拒绝高频词，使得低频词有更多出镜率，默认为0，即不进行亚采样
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;//词频高的词ran值低，容易被舍弃
          next_random = next_random * (unsigned long long)25214903917 + 11;
         //以1-ran的概率舍弃高频词
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word; // 句子里面存放的是词在词汇表里面的索引
        sentence_length++; // 当前处理的单词在当前句子中的位置
        if (sentence_length >= MAX_SENTENCE_LENGTH) break; // 如果句子长度超出最大长度则截断,超过1000个单词则截断。
      }
      sentence_position = 0; // 定位到句子头，表示当前单词在当前句中的index，起始值为0
    }
   // 如果当前线程处理的词数超过了它应该处理的最大值，那么开始新一轮迭代
   // 如果迭代数超过上限，则停止迭代
    if (feof(fi) || (word_count > train_words / num_threads)) { // 读到末尾或者数据超过最大处理值
      word_count_actual += word_count - last_word_count; // 更新word_count_actual，所有线程总共当前处理的词数
      local_iter--; // 迭代总次数减少一次
      if (local_iter == 0) break; // 迭代超过上限，停止迭代,只有这里才是跳出最外层循环的地方
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET); // 将文件读指针重新移到到此线程所处理词的开头
      continue; // 重新开始循环
    }
    word = sen[sentence_position]; // 取句子中的第一个单词，开始运行BP算法
    if (word == -1) continue; // 如果没有这个单词，则继续,有点疑问？？？
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;//初始化输入词向量
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;// 初始化累计误差项
// 生成一个[0, window-1]的随机数，用来确定|context(w)|窗口的实际宽度
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;// b的大小在[0, window-1]之间
   /**
   然后就开始训练了，先初始化了neu1和neu1e的值。并且确定了窗口的起始位 置，
   通过b = next_random % window来确定，理论上，我们在中心词左右都是取大小为 window个上下文词，
   但是在代码中，并不是保证左右都是window个，而是左边为(window - b)个，
   右边为(window + b)个，总数仍然是2 * window个*
   */
/********如果使用的是CBOW模型：输入是某单词周围窗口单词的词向量，来预测该中心单词本身*********/
    if (cbow) {  //CBOW模型训练
     // 输入层 -> 隐藏层
      cw = 0;
     // 这个循环主要是为了求neu1：输入词向量，在CBOW模型中是Context(x)中各个词的向量和
    // 一个词的窗口为[setence_position - window + b, sentence_position + window - b]
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {// 去除窗口的中心词，这是我们要预测的内容，仅仅提取上下文
        c = sentence_position - window + a;// 一个词的窗口为[setence_position - window + b, sentence_position + window - b]
        // 不符合条件，则继续取
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c]; // sen数组中存放的是句子中的每个词在词表中的索引
        if (last_word == -1) continue;// 不存在该词，则继续，有点疑惑？？？
        // 累加词对应的向量。双重循环下来就是窗口额定数量的词每一维对应的向量累加。  
        // 累加后neu1的维度依然是layer1_size。  
        // 从输入层过度到隐含层。
        // neu1：输入词向量，在CBOW模型中是Context(x)中各个词的向量和，在skip-gram模型中是中心词的词向量
       // real *neu1 = (real *)calloc(layer1_size, sizeof(real));
       // syn0 表示： 存储词典中每个词的词向量
       // real *syn0，syn0存储的是词表中每个词的词向量,源码中使用一个real(float)类型的一维数组表示，注意是一个一维数组！
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size]; // 计算窗口中词向量的和
// 注意syn0是一维数组，不是二维的，所以通过last_word * layer1_size来定位某个词对应的向量位置, last_word表示上下文中上一个词
        cw++; // 进入隐含层的词个数。 
      }
      if (cw) { // 有效次数大于1
      for (c = 0; c < layer1_size; c++) neu1[c] /= cw; // 归一化处理，求平均值， neu1是投影层的向量和
      // 注意hs模式下，syn1存的是非叶子节点对应的向量，并不是词汇表中的词对应的另一个向量；
      // 而negative模型下，syn1neg存的是词的另一个向量，需要注意
      // 如果采用分层softmax优化
      // 根据Haffman树上从根节点到当前词的叶节点的路径，遍历所有经过的中间节点
        if (hs) for (d = 0; d < vocab[word].codelen; d++) { // codelen是编码长度，word是当前词的索引
          f = 0;
          // 霍夫曼树中从根节点到该词的路径，存放路径上每个非叶结点的索引
          l2 = vocab[word].point[d] * layer1_size;// l2为当前遍历到的中间节点的向量在syn1中的起始位置，syn1 用于表示hs(hierarchical softmax)算法中霍夫曼编码树非叶结点的权重，syn1也是一个一维向量
          // 隐藏层 -> 输出层
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2]; // f为输入向量neu1与中间结点向量的内积
         // 检测f有没有超出Sigmoid函数表的范围
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
         // 如果没有超出范围则对f进行Sigmoid变换
         // sigmod函数， f=expTab[(int)((f+6)*1000/12)]  （计算出属于哪一份）
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];// f计算出来了
          // g是梯度和学习率的乘积
          // 注意！word2vec中将Haffman编码为1的节点定义为负类，而将编码为0的节点定义为正类
         // 即一个节点的label = 1 - d
          g = (1 - vocab[word].code[d] - f) * alpha;
          // 根据计算得到的修正量g和中间节点的向量更新累计误差
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // 根据计算得到的修正量g和输入向量更新中间节点的向量值
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // 负采样，暂时略过
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // 根据获得的的累计误差，更新context(w)中每个词的词向量
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];// 记录在词汇表中的位置
          if (last_word == -1) continue;// 一般不存在这种情况
          // 以上部分都与前面一致
          // neu1e存放的是累计误差向量,syn0存放的是词汇表中每个词汇的词向量
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  // skip-gram 模型，skip-gram其实没有投影层
      // 因为需要预测Context(w)中的每个词，因此需要循环2window - 2b + 1次遍历整个窗口
      // 每个上下文都要有一次循环
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {// 遍历时，略过中心单词
        c = sentence_position - window + a;
        // last_word为当前待预测的上下文单词
        // 不符合条件，则继续
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];// 取出该词在词汇表中的索引
        if（last_word ==-1）continue;// 一般不会出现这种情况
        l1 = last_word * layer1_size;// l1为当前单词的词向量在syn0中的起始位置,syn中存的是词汇表中词的向量
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;// 初始化累计误差
        // HIERARCHICAL SOFTMAX huffman树
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {//d代表公式里的l
          f = 0;// 初始化为0
        // l2为当前遍历到的中间节点的向量在syn1中的起始位置，syn1 用于表示hs(hierarchical softmax)算法中霍夫曼编码树非叶结点的权重，syn1也是一个一维向量
          l2 = vocab[word].point[d] * layer1_size;
          // 从隐藏层到输出层
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];// f为与当前上下文向量与中间结点向量的内积
          // 检测f有没有超出Sigmoid函数表的范围
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
         // 如果没有超出范围则对f进行Sigmoid变换
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//f计算出来了
         // 注意！这里用到了模型对称：p(u|w) = p(w|u)，其中w为中心词，u为context(w)中每个词
         // 也就是skip-gram虽然是给中心词预测上下文，真正训练的时候还是用上下文预测中心词
         // 与CBOW不同的是这里的u是单个词的词向量，而不是窗口向量之和
         // 算法流程基本和CBOW的hs一样
         // g是梯度和学习率的乘积
          g = (1 - vocab[word].code[d] - f) * alpha;
         // 计算累计误差值
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
         // 根据计算得到的修正量g和输入向量更新中间节点的向量值
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
         // 负采样，暂时略过
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // 处理完一个词，及时去更新他的词向量，用累计误差来更新。
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        // 每个窗口向量都要遍历一遍，遍历完成以后，读取下一个词。
      }
    }
        // 都是为了跳到外面的循环进入下一步
    sentence_position++;
        // 完成了一个词的训练，句子中位置往后移一个词
    if (sentence_position >= sentence_length) {// 处理完一句句子后，将句子长度置为零，进入循环
        // 重新读取句子并进行逐词计算,即读取下一个句子
      sentence_length = 0;// sentence_length 设置为0以后就
      continue;
    }
  }
  fclose(fi);// 关闭文件
  free(neu1);// 释放内存
  free(neu1e);// 释放内存
  pthread_exit(NULL);// 退出线程
}
```

## 5.4 ReadWordIndex（）从文件流中读取一个词，并返回这个词在词汇表中的位置

```c++
//构建词库的过程：从文件流中读取一个词，并返回这个词在词汇表中的位置
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);//读取一个词
  if (feof(fin)) return -1;
  return SearchVocab(word);//返回该词在词汇表中的位置
}
```

