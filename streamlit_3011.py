import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import RobustScaler

# URL của Google Sheets
sheet_id = '1A2EE7TrZrjULJkwHqdDXzoY5KOQkWAkcPxYxsC4B6rQ'
sheet_name = 'LogisticDataset'
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# Đọc dữ liệu từ Google Sheets
@st.cache
def load_data():
    return pd.read_csv(url)

# Hàm làm sạch dữ liệu
def clean_data(df):
    # Loại bỏ các cột không cần thiết
    df = df.T.drop_duplicates(keep='first').T
    df = df.drop(columns=['Customer Email', 'Product Description', 'Order Zipcode', 'Customer Zipcode',
                          'Product Image', 'Latitude', 'Longitude', 'Customer Fname', 'Customer Lname',
                          'Product Status', 'Category Id', 'Department Id', 'Customer Id', 'Order Id',
                          'Order Item Cardprod Id'], errors='ignore')

    # Chuyển đổi kiểu dữ liệu
    float_cols = ['Order Item Id', 'Order Item Quantity', 'Days for shipping (real)', 
                  'Days for shipment (scheduled)', 'Benefit per order', 'Sales per customer', 
                  'Order Item Discount', 'Order Item Discount Rate', 'Order Item Product Price', 
                  'Order Item Profit Ratio', 'Sales']
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')

    # Xử lý giá trị null
    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype == 'float':
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)

    return df

# Streamlit App
st.title("Logistics Data Analysis and Prediction App")

# Tải dữ liệu từ Google Sheets và hiển thị
try:
    df = load_data()
    st.write("### Dữ liệu từ Google Sheets:")
    st.dataframe(df.head())

    # Làm sạch dữ liệu
    cleaned_df = clean_data(df)
    st.write("### Dữ liệu sau khi làm sạch:")
    st.dataframe(cleaned_df.head())
    # Tạo cột Late_delivery_risk từ dữ liệu
    cleaned_df['Late_delivery_risk'] = (cleaned_df['Days for shipping (real)'] > cleaned_df['Days for shipment (scheduled)']).astype(int)

    # Trực quan hóa: Phân phối Days for Shipping (Real)
    st.write("### Phân phối số ngày giao hàng thực tế")
    fig, ax = plt.subplots()
    sns.histplot(cleaned_df['Days for shipping (real)'], kde=True, ax=ax)
    ax.set_title("Distribution of Days for Shipping (Real)")
    st.pyplot(fig)

    # Trực quan hóa khác
    st.write("### Market Distribution")
    fig, ax = plt.subplots()
    cleaned_df['Market'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), ax=ax)
    ax.set_title('Market Distribution')
    ax.set_ylabel('')
    st.pyplot(fig)

    st.write("### Customer Segment Distribution")
    fig, ax = plt.subplots()
    cleaned_df['Customer Segment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), ax=ax)
    ax.set_title('Customer Segment Distribution')
    ax.set_ylabel('')
    st.pyplot(fig)

    st.write("### Count of Each Category Name")
    category_counts = cleaned_df['Category Name'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='coolwarm', ax=ax)
    ax.set_title('Count of Each Category Name')
    ax.set_xlabel('Category Name')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.write("### Top 10 Most Common Products")
    product_counts = cleaned_df['Product Name'].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=product_counts.index, y=product_counts.values, palette='coolwarm', ax=ax)
    ax.set_title('Top 10 Most Common Products')
    ax.set_xlabel('Product Name')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.write("### Pair Plot of Continuous Variables")
    fig = sns.pairplot(cleaned_df[['Sales per customer', 'Benefit per order', 'Order Item Product Price']], diag_kind='kde', palette='coolwarm')
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    # Lọc bỏ các cột chỉ toàn giá trị null hoặc các cột có giá trị giống nhau
    numeric_df = cleaned_df.select_dtypes(include=['number']).loc[:, (cleaned_df.select_dtypes(include=['number']).nunique() > 1)]
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", ax=ax, annot_kws={"size": 8})
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    # Dự báo với mô hình giả lập
    st.sidebar.header("Nhập thông tin để dự đoán")
    days_for_shipping_real = st.sidebar.number_input("Days for Shipping (Real)", min_value=1, max_value=30, value=5)
    days_for_shipment_scheduled = st.sidebar.number_input("Days for Shipment (Scheduled)", min_value=1, max_value=30, value=5)
    sales_per_customer = st.sidebar.number_input("Sales per Customer", min_value=0, max_value=10000, value=500)
    benefit_per_order = st.sidebar.number_input("Benefit per Order", min_value=-500, max_value=1000, value=100)
    delivery_status = st.sidebar.selectbox(
    "Delivery Status",
    ['Advance shipping', 'Late delivery', 'Shipping on time', 'Shipping canceled']
)
    category_name = st.sidebar.selectbox(
    "Category Name",
    [
        'Sporting Goods', 'Cleats', 'Shop By Sport', "Women's Apparel", 'Electronics',
        'Boxing & MMA', 'Cardio Equipment', 'Trade-In', "Kids' Golf Clubs",
        'Hunting & Shooting', 'Baseball & Softball', "Men's Footwear",
        'Camping & Hiking', 'Consumer Electronics', 'Cameras ', 'Computers',
        'Basketball', 'Soccer', "Girls' Apparel", 'Accessories', "Women's Clothing",
        'Crafts', "Men's Clothing", 'Tennis & Racquet', 'Fitness Accessories',
        'As Seen on TV!', 'Golf Balls', 'Strength Training', "Children's Clothing",
        'Lacrosse', 'Baby ', 'Fishing', 'Books ', 'DVDs', 'CDs ', 'Garden', 'Hockey',
        'Pet Supplies', 'Health and Beauty', 'Music', 'Video Games', 'Golf Gloves',
        'Golf Bags & Carts', 'Golf Shoes', 'Golf Apparel', "Women's Golf Clubs",
        "Men's Golf Clubs", 'Toys', 'Water Sports', 'Indoor/Outdoor Games'
    ]
)
    customer_state = st.sidebar.selectbox(
    "Customer State",
    [
        'PR', 'CA', 'NY', 'FL', 'MA', 'IL', 'MT', 'PA', 'MI', 'TX', 'DE', 'GA',
        'MD', 'OH', 'HI', 'NJ', 'WI', 'AZ', 'CO', 'MN', 'NC', 'NM', 'OR', 'SC',
        'VA', 'UT', 'WA', 'KY', 'WV', 'RI', 'CT', 'LA', 'TN', 'DC', 'ND', 'MO',
        'IN', 'ID', 'NV', 'KS', 'AR', 'OK', 'AL', 'IA'
    ]
)
    order_country = st.sidebar.selectbox(
    "Order Country",
    [
        'Indonesia', 'India', 'Australia', 'China', 'Japón', 'Corea del Sur',
        'Singapur', 'Turquía', 'Mongolia', 'Estados Unidos', 'Nigeria',
        'República Democrática del Congo', 'Senegal', 'Marruecos', 'Alemania',
        'Francia', 'Países Bajos', 'Reino Unido', 'Guatemala', 'El Salvador',
        'Panamá', 'República Dominicana', 'Venezuela', 'Colombia', 'Honduras',
        'Brasil', 'México', 'Uruguay', 'Argentina', 'Cuba', 'Perú', 'Nicaragua',
        'Ecuador', 'Angola', 'Sudán', 'Somalia', 'Costa de Marfil', 'Egipto',
        'Italia', 'España', 'Suecia', 'Austria', 'Canada', 'Madagascar',
        'Argelia', 'Liberia', 'Zambia', 'Níger', 'SudAfrica', 'Mozambique',
        'Tanzania', 'Ruanda', 'Israel', 'Nueva Zelanda', 'Bangladés',
        'Tailandia', 'Irak', 'Arabia Saudí', 'Filipinas', 'Kazajistán', 'Irán',
        'Myanmar (Birmania)', 'Uzbekistán', 'Benín', 'Camerún', 'Kenia', 'Togo',
        'Ucrania', 'Polonia', 'Portugal', 'Rumania', 'Trinidad y Tobago',
        'Afganistán', 'Pakistán', 'Vietnam', 'Malasia', 'Finlandia', 'Rusia',
        'Irlanda', 'Noruega', 'Eslovaquia', 'Bélgica', 'Bolivia', 'Chile',
        'Jamaica', 'Yemen', 'Ghana', 'Guinea', 'Etiopía', 'Bulgaria',
        'Kirguistán', 'Georgia', 'Nepal', 'Emiratos Árabes Unidos', 'Camboya',
        'Uganda', 'Lesoto', 'Lituania', 'Suiza', 'Hungría', 'Dinamarca',
        'Haití', 'Bielorrusia', 'Croacia', 'Laos', 'Baréin', 'Macedonia',
        'República Checa', 'Sri Lanka', 'Zimbabue', 'Eritrea', 'Burkina Faso',
        'Costa Rica', 'Libia', 'Barbados', 'Tayikistán', 'Siria', 'Guadalupe',
        'Papúa Nueva Guinea', 'Azerbaiyán', 'Turkmenistán', 'Paraguay',
        'Jordania', 'Hong Kong', 'Martinica', 'Moldavia', 'Qatar', 'Mali',
        'Albania', 'República del Congo', 'Bosnia y Herzegovina', 'Omán',
        'Túnez', 'Sierra Leona', 'Yibuti', 'Burundi', 'Montenegro', 'Gabón',
        'Sudán del Sur', 'Luxemburgo', 'Namibia', 'Mauritania', 'Grecia',
        'Suazilandia', 'Guyana', 'Guayana Francesa',
        'República Centroafricana', 'Taiwán', 'Estonia', 'Líbano', 'Chipre',
        'Guinea-Bissau', 'Surinam', 'Belice', 'Eslovenia', 'República de Gambia',
        'Botsuana', 'Armenia', 'Guinea Ecuatorial', 'Kuwait', 'Bután', 'Chad',
        'Serbia', 'Sáhara Occidental'
    ]
)
    order_status = st.sidebar.selectbox(
    "Order Status",
    ['COMPLETE', 'PENDING', 'CLOSED', 'PENDING_PAYMENT', 'CANCELED', 
     'PROCESSING', 'SUSPECTED_FRAUD', 'ON_HOLD', 'PAYMENT_REVIEW']
)
    order_region = st.sidebar.selectbox(
    "Order Region",
    [
        'Southeast Asia', 'South Asia', 'Oceania', 'Eastern Asia', 'West Asia',
        'West of USA ', 'US Center ', 'West Africa', 'Central Africa', 
        'North Africa', 'Western Europe', 'Northern Europe', 'Central America',
        'Caribbean', 'South America', 'East Africa', 'Southern Europe', 
        'East of USA', 'Canada', 'Southern Africa', 'Central Asia', 
        'Eastern Europe', 'South of  USA '
    ]
)
    market = st.sidebar.selectbox(
    "Market",
    ['Pacific Asia', 'USCA', 'Africa', 'Europe', 'LATAM']
)
    # Mô hình giả lập
    features = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Sales per customer', 'Benefit per order']
    clf = RandomForestClassifier()
    reg = RandomForestRegressor()
    clf.fit(cleaned_df[features], cleaned_df['Late_delivery_risk'])
    reg.fit(cleaned_df[features], cleaned_df['Sales'])

    # Dự báo
    user_input = np.array([[days_for_shipping_real, days_for_shipment_scheduled, sales_per_customer, benefit_per_order]])
    delivery_risk = clf.predict(user_input)[0]
    predicted_sales = reg.predict(user_input)[0]

    # Hiển thị kết quả dự đoán
    st.write("### Kết quả dự đoán")
    st.write(f"**Khả năng giao hàng trễ:** {'Có' if delivery_risk == 1 else 'Không'}")
    st.write(f"**Dự báo doanh số (Sales):** ${predicted_sales:.2f}")

except Exception as e:
    st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")
